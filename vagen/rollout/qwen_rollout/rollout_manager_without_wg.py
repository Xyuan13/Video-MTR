from typing import List, Union, Optional, Dict
import copy
from collections import defaultdict
import torch
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin
from dataclasses import dataclass, field
import PIL
import re
import logging
import os
import psutil

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import process_image, collate_fn
import vagen.env
from vagen.env import REGISTERED_ENV

from vllm import LLM, SamplingParams


def setup_rollout_logger():
    """Setup logger that writes to wandb debug.log if available, otherwise to console"""
    logger = logging.getLogger('rollout_manager')
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    # Prevent propagation to parent loggers to avoid duplicate messages
    logger.propagate = False
    
    # Create formatter matching wandb's format
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(threadName)s:%(process)d [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S,%f'
    )
    
    # Try to find wandb debug.log file
    wandb_log_file = None
    try:
        import wandb
        if wandb.run is not None:
            # Get the current run's log directory
            run_dir = wandb.run.dir
            if run_dir:
                # wandb.run.dir points to files/, we need logs/debug.log
                log_dir = os.path.join(os.path.dirname(run_dir), 'logs')
                wandb_log_file = os.path.join(log_dir, 'debug.log')
                
                # Verify the file exists and is writable
                if not os.path.exists(wandb_log_file):
                    wandb_log_file = None
    except ImportError:
        pass
    
    if wandb_log_file:
        # Add file handler for wandb debug.log
        file_handler = logging.FileHandler(wandb_log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"[INFO] Rollout logger writing to wandb debug.log: {wandb_log_file}")
    else:
        # Add console handler only if no wandb file is available
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        print("[INFO] Rollout logger writing to console (no wandb detected)")
    
    return logger

class QwenVLRolloutManager_WithoutWG():
    def __init__(self,
                 config,
                 llm,
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 sampling_params = None,
                 ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.sampling_params = sampling_params

        self.recorder= None # defaultdict(list) env_id:record
        self.envs = None # dict env_id:EnvInterface
        self.env_states = None # dict
        self.batch_idx_to_env_id = None # dict
        
        # Setup logger that writes to wandb debug.log
        self.logger = setup_rollout_logger()

    @torch.no_grad()
    def _handle_special_tokens(self, llm_raw_response: str, prep_for_loss_mask: bool) -> str:
        """
        1. Filter out special tokens: <image> and special tokens marking environment observation in the llm generated response
        2. prep_for_loss_mask: if true, add special tokens to the beginning and end of the response if compute_loss_mask is True
        """
        llm_raw_response = llm_raw_response.replace('<image>', '')
        if prep_for_loss_mask:
            # filtering special tokens for llm_raw_response, then adding them to the beginning and end of the response for loss mask computation
            sptk_b = self.config.special_token_for_loss_mask[0]
            sptk_e = self.config.special_token_for_loss_mask[1]
            llm_raw_response = llm_raw_response.replace(sptk_b, '')
            llm_raw_response = llm_raw_response.replace(sptk_e, '')
            llm_raw_response = sptk_b + llm_raw_response + sptk_e
        return llm_raw_response
    
    @torch.no_grad()
    def _handle_multi_modal_data(
            self, 
            prompt_template: str, 
            row_dict: Dict,
            image_data: List[PIL.Image.Image],
            do_embedding: bool = True,
        ) -> str:
        """Handle multi-modal data in the prompt template

        - For do_embedding=False(vllm), replace <image> with <|vision_start|><|image_pad|><|vision_end|> -> raw_prompt
        - For do_embedding=True, replace <image> with <|vision_start|>{image_token}<|vision_end|> -> prompt_template
            - where {image_token} is the length of image embedding
        """
        assert len(image_data) == prompt_template.count('<image>'), 'Number of images does not match number of <image> in the prompt template'
        raw_prompt = prompt_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        row_dict['multi_modal_data'] = {'image': image_data}
        image_grid_thw = None
        if do_embedding:
            image_inputs = self.processor.image_processor(image_data, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            #print(f"[DEBUG] do embedding, number of image_data in rollout: {len(image_data)}")
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            while '<image>' in prompt_template:
                prompt_template = prompt_template.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                    '<|vision_end|>',
                    1,
                )
                index += 1

            prompt_template = prompt_template.replace('<|placeholder|>',
                                                        self.processor.image_token)
            # print(f"[DEBUG] number of image_data in final trajectory: {len(image_data)}")
            # number_of_image_tokens=prompt_template.count(self.processor.image_token)
            # print(f"[DEBUG] number_of_image_tokens: {number_of_image_tokens}")
        return prompt_template, row_dict, image_grid_thw, raw_prompt
    
    @torch.no_grad()
    def _compute_loss_mask(self, input_ids, attention_mask):
        """
        Compute loss mask for the input ids and attention mask
        We only do loss for the tokens in input_ids that are wrapped by special tokens (by defualt they're <|box_start|> and <|box_end|>)
        
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
    
        Returns:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            loss_mask: (batch_size, seq_len) # e.g. 0000|1111|0000|11111|000|1111
            end_of_response_position_mask: (batch_size, seq_len) # e.g. 0000|0001|0000|00001|000|0001 given the end of sequence mask, mark the position of the last token in the response
        
        - There will be different stratgy to handel special tokens in the input_ids
        - 1. remove them, in this case we need to fill the hole by adding pad in the right and shift the sequence left
        - 2. keep them, attention mask will be 0 for them
        - 3. Replace them with pad token
    
        Let's use the 3rd strategy for now
        Compute loss mask for the input ids and attention mask by:
        1. Removing special tokens
        2. Adding padding on the right
        3. Shifting the sequence left
        """

        # Get token IDs for special tokens and pad token
        sptk_b = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[0])
        sptk_e = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[1])
        pad_token_id = self.tokenizer.pad_token_id

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Initialize output tensors with same shape as inputs
        new_input_ids = input_ids.clone()
        new_attention_mask = attention_mask.clone()
        loss_mask = torch.zeros_like(new_attention_mask)
        new_loss_mask = torch.zeros_like(new_attention_mask)
        end_of_response_position_mask = torch.zeros_like(new_attention_mask)
        new_end_of_response_position_mask = torch.zeros_like(new_attention_mask)
        # Process each example in the batch
        for b in range(batch_size):
            # Count right padding tokens using attention mask
            right_pad_tokens = (new_input_ids[b] == pad_token_id).sum().item()
            
            # Assert that initial padding tokens have attention mask of 0
            if not torch.all(attention_mask[b, -right_pad_tokens:] == 0):
                print("[DEBUG]: right padding tokens must have attention mask of 0")
            
            # Find special token indices
            sptk_b_indices = (input_ids[b] == sptk_b).nonzero().flatten()
            sptk_e_indices = (input_ids[b] == sptk_e).nonzero().flatten()
            
            # Create a mask for tokens that should compute loss
            hole_pos=[] # initialize holes position list with last padding token position
            for start_pos, end_pos in zip(sptk_b_indices, sptk_e_indices):
                loss_mask[b][start_pos+1:end_pos] = 1
                end_of_response_position_mask[b][end_pos-1] = 1
                hole_pos.append(start_pos.item())
                hole_pos.append(end_pos.item())
            hole_pos.append(seq_len-right_pad_tokens)
            # assert new_input_ids[b][seq_len-right_pad_tokens]==pad_token_id
            if not torch.all(new_input_ids[b][seq_len-right_pad_tokens:] == pad_token_id):
                print("[DEBUG]: right padding tokens must be pad token")
            
            # shift right to fill the wholes
            holes_to_fill=1
            for i in range(0,len(hole_pos)-1):
                start_pos = hole_pos[i]
                end_pos = hole_pos[i+1]
                new_loss_mask[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=loss_mask[b,start_pos+1:end_pos]
                new_end_of_response_position_mask[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=end_of_response_position_mask[b,start_pos+1:end_pos]
                new_input_ids[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=input_ids[b,start_pos+1:end_pos]
                new_attention_mask[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=attention_mask[b,start_pos+1:end_pos]
                holes_to_fill+=1

            valid_tokens = seq_len-right_pad_tokens-len(hole_pos)+1 # the number of non-special tokens and non-padding tokens
            new_loss_mask[b][valid_tokens:]=0
            new_input_ids[b][valid_tokens:]=pad_token_id
            new_attention_mask[b][valid_tokens:]=0
        
        # -- Statistics for loss mask --
        for b in range(batch_size):
            mask_length = new_loss_mask[b].shape[0]
            mask_nonzero_count = torch.sum(new_loss_mask[b]).item()
            # print(f"Loss mask stats - batch {b}: length={mask_length}, nonzero={mask_nonzero_count}, ratio={mask_nonzero_count/mask_length:.3f}")
        #--------------------------------
        
        return new_input_ids, new_attention_mask, new_loss_mask, new_end_of_response_position_mask
    
    @torch.no_grad()
    def reset(self, env_configs):
        """
        Reset environments based on provided configurations, reusing environments when possible.
        - For env with same config and env_name, reuse the same environment (reset)
        - For env with different config or env_name, close the old environment and create a new one
        - Reset the recorder
        
        Args:
            env_configs: List of environment configurations containing env_name, config, and seed
        
        Returns:
            Initial observations and info from all environments
        """
        # Step 1: Sort environments into buckets by env_name and config
        # Try to reuse environemnts with the same config and env_name
        
        env_buckets = defaultdict(set)
        new_envs = {}
        
        if self.envs is None:
            self.envs = {}
            
        for env_id, env in self.envs.items():
            env_config_id = env.config.config_id()
            bucket_key = env_config_id
            env_buckets[bucket_key].add(env_id)

        if 'split' in env_configs[0]:
            print(f"[Debug] RolloutManager: Reset {len(env_configs)} envs with split '{env_configs[0]['split']}'")
            
        for i, cfg in enumerate(env_configs):
            env_id = i
            env_name = cfg["env_name"]
            env_config = cfg["env_config"]
            seed = cfg["seed"]
            
            # Create bucket key
            config_instance= REGISTERED_ENV[env_name]["config_cls"](**env_config)
            env_config_id = config_instance.config_id()
            bucket_key = env_config_id
            
            # Check if we have an available environment with the same config
            if bucket_key in env_buckets and env_buckets[bucket_key]:
                old_env_id = env_buckets[bucket_key].pop()
                new_envs[env_id] = {
                    "env_instance":self.envs[old_env_id],
                    "seed":seed,
                }
            else:
                # don't initialize the environment here, close unused environments first
                new_envs[env_id] = {
                    "env_cls":REGISTERED_ENV[env_name]["env_cls"],
                    "seed":seed,
                    "config_instance":config_instance,
                }
        
        # Close unused environments
        for bucket_key, env_ids in env_buckets.items():
            for env_id in env_ids:
                self.envs[env_id].close()
                del self.envs[env_id]

        
        # Step 2: Reset environments and collect observations/info
        
        if self.recorder is not None:
            del self.recorder
        self.recorder = defaultdict(list)
        initial_obs = {}
        initial_info = {}
        for env_id, env_info in new_envs.items():
            if "env_instance" in env_info:
                self.envs[env_id] = env_info["env_instance"]
            else:
                assert "env_cls" in env_info
                self.envs[env_id] = env_info["env_cls"](env_info["config_instance"])
            
            # ADD: Pass extra_info to environment reset if available
            env_config = env_configs[env_id]
            if hasattr(self.envs[env_id].reset, '__code__') and 'extra_info' in self.envs[env_id].reset.__code__.co_varnames:
                obs, info = self.envs[env_id].reset(env_info["seed"], extra_info=env_config)
                if 'split' in env_config:
                    #print(f"[Debug] RolloutManager: Reset env {env_id} with split '{env_config['split']}'")
                    pass
            else:
                obs, info = self.envs[env_id].reset(env_info["seed"])
            
            initial_obs[env_id] = obs
            initial_info[env_id] = info
            self.record(
                env_id, 
                obs=obs, 
                reward=0, 
                done=False, 
                info=info
            )
        
        self.env_states = {env_id: {'step': 0, 'done': False,'metrics':{"turn_metrics":defaultdict(list),"traj_metrics":{}}} for env_id in self.envs}
        
        return initial_obs, initial_info
    
    @torch.no_grad()
    def record(self, env_id, obs, reward, done, info):
        """
        Record each step's obs, info, done, reward,
        Please include "llm_raw_response" in info # it will be decoded by rollout manager and pass to env, then should pass back
        """
        # Create a record entry for this step
        assert obs is not None, "obs cannot be None"
        assert info is not None, "info cannot be None"
        assert isinstance(reward, (int, float)), "reward must be a number"
        assert isinstance(done, bool), "done must be a boolean"
        record_entry = {
            'env_id': env_id,
            'done': done,
            'reward': reward,
            'info': info,
            'obs_str': obs['obs_str'],
        }
        image_placeholder = self.envs[env_id].config.get('image_placeholder', "<image>")
        if 'multi_modal_data' in obs:
            if image_placeholder in obs['multi_modal_data']:
                #record_entry['image_data'] = [process_image(image) for image in obs['multi_modal_data'][image_placeholder]]
                # Modify: directly use the image data from obs
                record_entry['image_data'] = obs['multi_modal_data'][image_placeholder]
        self.recorder[env_id].append(record_entry)

        # Log the llm_raw_response
        #self.logger.debug(f"|| record: env_id={env_id} \n || record_entry['obs_str']={record_entry['obs_str']}\n || record_entry['info']={record_entry['info']}")

    @torch.no_grad()
    def _single_recording_to_prompt(self,
                            recording: List[Dict], 
                            step: int, 
                            window_size: int = None,
                            is_final: bool = False,
                            prep_for_loss_mask: bool = False,
        ):
        """
        Given a recording, generate the prompt for MLLM
        Chat: Sys -> |InitUser| -> |Assistant, User| -> |Assistant, User| -> ... -> |Assistant, User Final|

        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate prompt for
            window_size: Number of past steps to include in the context
            is_final: Whether the prompt is for the final step 
                - if True, the end of the chat is from the last assistant's response
            prep_for_loss_mask: whether to use special token to wrap llm response
            
        Returns:
            dict: prompt_with_chat_template : str, image_data: list of images, reward: list of reward
        """
        
        assert step >= 0
        start_step = max(0, step - window_size) if window_size is not None else 0
        end_step = step
        assert len(recording) >= end_step + 1, 'History length is not enough'
        history = recording[start_step: end_step + 1]
        rewards=[]
        chat = []
        
        env_id = history[0]['env_id']
        chat.append({"role": "system", "content": self.envs[env_id].system_prompt()})

        image_data=[]
        for i, record in enumerate(history):
            if i>0:
                llm_raw_response = record['info']['llm_raw_response']
                filtered_llm_raw_response = self._handle_special_tokens(llm_raw_response, prep_for_loss_mask=prep_for_loss_mask)
                chat.append({"role": "assistant", "content": filtered_llm_raw_response})
                rewards.append(record['reward'])
            if i<len(history)-1 or not is_final:
                chat.append({"role": "user", "content": record['obs_str']})
                if 'image_data' in record:
                    for img in record['image_data']:
                        image_data.append(img)
            
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=(not is_final), tokenize=False)
        if is_final: # NOTE hard coded
            assert prompt_with_chat_template[-1] == '\n', f"The last token should be new line token, got {prompt_with_chat_template[-1]}"
            prompt_with_chat_template = prompt_with_chat_template[:-1] # remove the last in token
        # switch box_end and im_end so that the model can learn to generate <|im_end|>
        prompt_with_chat_template = prompt_with_chat_template.replace(
            f'{self.config.special_token_for_loss_mask[1]}{self.tokenizer.eos_token}',
            f'{self.tokenizer.eos_token}{self.config.special_token_for_loss_mask[1]}')

        # Log the prompt generation details
        #self.logger.debug(f"single_recording_to_prompt: env_id={env_id}, step={step}, window_size={window_size}, len(image_data)={len(image_data)}")
        # self.logger.debug(f"- prompt_with_chat_template: {prompt_with_chat_template}")

        # Add: Calculate token count for prompt_with_chat_template
        prompt_token_count = len(self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False))

        return {
            "prompt": prompt_with_chat_template,
            "image_data": image_data,
            "rewards": rewards,
            "prompt_token_count": prompt_token_count,
        }
    
    @torch.no_grad()
    def _generate_input_for_rollout(
            self, 
            recording: List[Dict], 
            step: int, 
            window_size: int = None,
        ):
        """
        Given a recording, generate the input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute
        """
        rst=self._single_recording_to_prompt(recording, step, window_size, is_final=False, prep_for_loss_mask=False)
        prompt_with_chat_template=rst['prompt']
        image_data=rst['image_data']        
        has_images = len(image_data) > 0        

        #print(f"[Debug] _generate_input_for_rollout: \n - prompt_with_chat_template: {prompt_with_chat_template}")

        row_dict = {}
        if has_images:  # expand image token
            prompt_with_chat_template, row_dict, _, raw_prompt = self._handle_multi_modal_data(
                prompt_with_chat_template, row_dict, image_data, do_embedding=False)
            
            # print the number of images
            if step > 0:
                print(f"[Debug] _generate_input_for_rollout, step_{step}: number of images: {len(image_data)}")
        else:
            raw_prompt = prompt_with_chat_template

        # use random input_ids and attention_mask for vllm only takes raw_prompt_ids as input when generating sequences
        # TODO check if this is correct
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['input_ids'] = torch.tensor([0], dtype=torch.long)
        row_dict['attention_mask'] = torch.tensor([0], dtype=torch.long)
        row_dict['position_ids'] = torch.tensor([0], dtype=torch.long)

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        # print the number of tokens in the prompt
        #print(f"[Debug] _generate_input_for_rollout: prompt_token_count: {rst['prompt_token_count']}")

        return row_dict


    @torch.no_grad()
    def _generate_input_for_uptate(
            self, 
            recording: List[Dict], 
            step: int, 
            window_size: int = None,
        ):
        """
        Given a recording, generate the final input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute

        """



        # handle prompt, prompt=pad_token since we now have everything in response and compute a loss mask for them
        prompt_with_chat_template=self.tokenizer.pad_token 
        
        # handle response
        response_rst=self._single_recording_to_prompt(recording, step, window_size, is_final=True, prep_for_loss_mask=True)
        response_with_chat_template=response_rst['prompt']
        image_data=response_rst['image_data']
        rewards=response_rst['rewards']
       
        has_images = len(image_data) > 0
        row_dict = {}
        
        # Calculate pre-padding token counts for debugging
        pre_padding_prompt_token_count = len(self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False))
        pre_padding_response_token_count = len(self.tokenizer.encode(response_with_chat_template, add_special_tokens=False))
        
        if has_images:  # expand image token
            response_with_chat_template, row_dict, image_grid_thw, _ = self._handle_multi_modal_data(
                response_with_chat_template, row_dict, image_data, do_embedding=True)
            # Update response token count after multi-modal processing
            post_multimodal_response_token_count = len(self.tokenizer.encode(response_with_chat_template, add_special_tokens=False))
        else:
            post_multimodal_response_token_count = pre_padding_response_token_count

        
        input_ids_response, attention_mask_response = verl_F.tokenize_and_postprocess_data(prompt=response_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.config.max_trajectory_length-1, # -1 for the prompt padding token
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=False,
                                                                         truncation=self.config.truncation)
        input_ids_prompt, attention_mask_prompt = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=1,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.config.truncation)
        attention_mask_prompt=torch.zeros_like(input_ids_prompt) # All prompt will be masked
        
        
        input_ids_response, attention_mask_response, loss_mask_response,end_of_response_position_mask_response = self._compute_loss_mask(input_ids_response, attention_mask_response)
        
        input_ids_prompt=input_ids_prompt[0]
        attention_mask_prompt=attention_mask_prompt[0]
        loss_mask_prompt = torch.zeros_like(attention_mask_prompt)
        end_of_response_position_mask_prompt = torch.zeros_like(attention_mask_prompt)
        
        input_ids_response=input_ids_response[0]
        attention_mask_response=attention_mask_response[0]
        loss_mask_response=loss_mask_response[0]
        end_of_response_position_mask_response=end_of_response_position_mask_response[0]
        
    
        
        loss_mask = torch.cat([loss_mask_prompt, loss_mask_response], dim=-1)
        end_of_response_position_mask = torch.cat([end_of_response_position_mask_prompt, end_of_response_position_mask_response], dim=-1)
        input_ids = torch.cat([input_ids_prompt, input_ids_response], dim=-1)
        attention_mask = torch.cat([attention_mask_prompt, attention_mask_response], dim=-1)

        
        
        position_ids_prompt = compute_position_id_with_mask(attention_mask_prompt)
        # if self.image_key in row_dict:
        if has_images:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids_response = get_rope_index(
                self.processor,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids_response,
                attention_mask=attention_mask_response,
            )  # (3, seq_len)
            position_ids_prompt=position_ids_prompt.view(1, -1).expand(3, -1)
        else:
            response_length = input_ids_response.shape[0]
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids_prompt.device)
            position_ids_response = position_ids_prompt[-1:] + delta_position_id
        
        if self.config.use_multi_turn_reward:
            reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
            multi_turn_token_level_rewards = torch.zeros_like(end_of_response_position_mask, dtype=torch.float)
            assert len(reward_positions) == len(rewards), "Number of rewards does not match number of reward positions"
            print(f"[Debug] generate_input_for_uptate: final rewards({len(rewards)}): {rewards}")

            for idx,reward in enumerate(rewards):
                multi_turn_token_level_rewards[reward_positions[idx]] = reward
            row_dict["multi_turn_token_level_rewards"] = multi_turn_token_level_rewards # (seq_len,) 

        if self.config.use_loss_mask:
            # -- Final loss mask statistics --
            final_mask_length = loss_mask.shape[0]
            final_mask_nonzero = torch.sum(loss_mask).item()
            # print(f"[Debug] Final loss mask stats: length={final_mask_length}, nonzero={final_mask_nonzero}, ratio={final_mask_nonzero/final_mask_length:.3f}")
            #--------------------------------
            row_dict['loss_mask'] = loss_mask
        if self.config.use_gae_mask:
            row_dict['gae_mask'] = loss_mask
        row_dict["end_of_response_position_mask"] = end_of_response_position_mask # 
        position_ids = torch.cat([position_ids_prompt, position_ids_response], dim=-1)
        row_dict['prompts'] = input_ids_prompt
        row_dict['responses'] = input_ids_response
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        
        # Log the key metrics with pre/post padding comparison
        response_token_count = len(input_ids_response)
        final_input_ids_length = len(input_ids)
        final_response_length = len(input_ids_response)
        final_prompt_length = len(input_ids_prompt)
        
        # Calculate multimodal expansion (only show this)
        multimodal_expansion = post_multimodal_response_token_count - pre_padding_response_token_count if has_images else 0
        # Calculate padding expansion
        padding_expansion = final_response_length - post_multimodal_response_token_count
        
        # print(f"[Debug] rollout_manager.py tokens_stats: "
        #                  f"response({pre_padding_response_token_count}), "
        #                  f"add_mm({post_multimodal_response_token_count}), "
        #                  f"final={final_response_length}), "
        #                  f"mm={multimodal_expansion:+d}, padding={padding_expansion:+d}")
        
        return row_dict

    @torch.no_grad()
    def _get_answer_binded_stepwise_tool_reward(self, batch):
        """
        Get the answer-binded stepwise tool reward for the batch
        """
        answer_reward_threshold = 0.5
        for row_dict in batch:
            # torch.Tensor type, shape (seq_len,)
            multi_turn_rewards = row_dict['multi_turn_token_level_rewards']
            # Find indices of elements greater than 0
            positive_indices = torch.nonzero(multi_turn_rewards > 0, as_tuple=True)[0]

            # Extract rewards before modification
            original_rewards = multi_turn_rewards[positive_indices].tolist()

            if len(positive_indices) < 2: # no tool used, no need to bind the answer reward
                continue 

            # answer index is the last element of positive_indices
            answer_index = positive_indices[-1]
            
            # bind the answer: only get the tool reward when the answer is true
            if multi_turn_rewards[answer_index]  < answer_reward_threshold:
                # remove the tool reward
                tool_indices = positive_indices[:-1]

                # find out the corret tool used reward , which > 0.5
                correct_tool_indices = torch.nonzero(row_dict['multi_turn_token_level_rewards'][tool_indices] > 0.5, as_tuple=True)[0]
                row_dict['multi_turn_token_level_rewards'][tool_indices[correct_tool_indices]] -= 0.5

                # Extract rewards after modification
                updated_multi_turn_rewards = row_dict['multi_turn_token_level_rewards']
                updated_rewards = updated_multi_turn_rewards[positive_indices].tolist()

                print(f"[Debug:Answer-binded] Original rewards: {original_rewards}")
                print(f"[Debug:Answer-binded] Updated rewards: {updated_rewards}")
                print(f"---------------------------------------------------------------")
                # print the status of cpu memory
                print(f"[Debug:Answer-binded] CPU memory: {psutil.virtual_memory().percent}%")


            #if answer_is_true:
        return batch

    @torch.no_grad()
    def _compute_batch_tool_usage_reward(self, batch):
        """
        Compute the tool usage reward for the batch
        """
        
        non_zero_count = 0
        tool_used_count = 0

        TOOL_USED_RATIO_THRESHOLD = 0.5 # if use tool less than 50% of the multi-turn, then give extra reward
        TOOL_USED_REWARD = 0.5
    
        for row_dict in batch:
            # torch.Tensor type, shape (seq_len,)
            multi_turn_rewards = row_dict['multi_turn_token_level_rewards']
            positive_reward_count = torch.sum(multi_turn_rewards > 0).item()
    
                
            if positive_reward_count > 0:
                non_zero_count += 1                 
                # check if use tool in the multi-turn
                if positive_reward_count > 1:
                    tool_used_count += 1
          
        # compute the tool used ratio
        tool_used_ratio = tool_used_count / non_zero_count if non_zero_count > 0 else 0
        print(f"[Debug] _compute_batch_tool_usage_reward: batch_size={len(batch)}, non_zero={non_zero_count}, tool_used={tool_used_count}, ratio={tool_used_ratio:.2f}")

        if tool_used_ratio < TOOL_USED_RATIO_THRESHOLD:
            print(f"[Debug] Tool usage ratio {tool_used_ratio:.2f} < threshold {TOOL_USED_RATIO_THRESHOLD}, applying tool usage reward")
            
            for row_dict in batch:
                multi_turn_rewards = row_dict['multi_turn_token_level_rewards']
                # Find indices of elements greater than 0
                positive_indices = torch.nonzero(multi_turn_rewards > 0, as_tuple=True)[0]
                     
                if len(positive_indices) > 1:
                    # Add the extra tool used reward(exclude the final action)
                    positive_indices_for_reward = positive_indices[:-1]
                     
                    # Extract rewards before modification
                    original_rewards = multi_turn_rewards[positive_indices].tolist()
                    print(f"[Debug] Original rewards: {original_rewards}")
                     
                    # Apply extra reward
                    row_dict['multi_turn_token_level_rewards'][positive_indices_for_reward] += TOOL_USED_REWARD
                    if tool_used_ratio < 0.1: # extra reward for cold start
                        row_dict['multi_turn_token_level_rewards'][positive_indices_for_reward] += TOOL_USED_REWARD

                     
                    # Extract rewards after modification
                    updated_multi_turn_rewards = row_dict['multi_turn_token_level_rewards']
                    updated_rewards = updated_multi_turn_rewards[positive_indices].tolist()
                    print(f"[Debug] Updated rewards: {updated_rewards}")
                        
        return batch

    @torch.no_grad()
    def generate_batch_for_rollout(self, step, window_size):
        """
        Generate a batch of data for the current step
        
        Args:
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - None if no data is available (all environments are done)
        """
        batch = []
        self.batch_idx_to_env_id = {}
        batch_idx = 0
        for env_id in self.envs.keys():
            if self.env_states[env_id]['done']:
                continue
            
            batch_row_dict = self._generate_input_for_rollout(self.recorder[env_id], step, window_size)
            batch.append(batch_row_dict)
            self.batch_idx_to_env_id[batch_idx] = env_id
            batch_idx += 1
        if not batch:
            return None

        if len(batch) % self.config.n_gpus_per_node != 0:
            # Pad the batch to make it divisible by n_gpus_per_node
            while len(batch) % self.config.n_gpus_per_node != 0:
                # do we need to use copy or not here?
                batch.append(batch[-1].copy())

        return collate_fn(batch)
    
    @torch.no_grad()
    def rollout_loop(self):
        """
        Step the environment and record the results
        
        Returns:
            Dictionary containing the results of the step
        """
        for step in range(self.config.max_turns):
            input_batch_dict = self.generate_batch_for_rollout(step, self.config.window_size)
            if input_batch_dict is None:
                break
            input_batch = DataProto.from_single_dict(input_batch_dict)
            if 'multi_modal_data' in input_batch.non_tensor_batch.keys():
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                )
            else:
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            # transform raw_prompt_ids to list instead of numpy array
            # The reason is that when constructing raw_prompt_ids, if the all the list share the same length
            # Numpy array will automatically transfer list to numpy array.
            raw_prompt_ids = gen_batch.non_tensor_batch['raw_prompt_ids']
            raw_prompt_ids_array = np.ndarray(shape=(len(raw_prompt_ids),), dtype=object)
            for i in range(len(raw_prompt_ids)):
                if isinstance(raw_prompt_ids[i],list):
                    raw_prompt_ids_array[i] = raw_prompt_ids[i]
                else:
                    raw_prompt_ids_array[i] = raw_prompt_ids[i].tolist()
            gen_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids_array
            
            
            # -----------------------------------------------------------------------
            #output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)
            # responses_str = self.tokenizer.batch_decode(
            #     output_batch.batch['responses'], 
            #     skip_special_tokens=True
            # ) # seems here will remove special token like "<|im_end|>"

            llm_inputs = []
            for i in range(len(gen_batch)):
                llm_inputs.append({
                    "prompt": gen_batch.non_tensor_batch['raw_prompt_ids'][i],
                    "multi_modal_data": gen_batch.non_tensor_batch['multi_modal_data'][i],
                    #"mm_processor_kwargs": gen_batch.non_tensor_batch['mm_processor_kwargs'][i],
                })
                
            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
            #batch_output_text = [out.outputs[0].text for out in outputs]
            
            responses_str = [out.outputs[0].text for out in outputs]
            # --------------------------------------------------------------------------


            for batch_idx, env_id in self.batch_idx_to_env_id.items(): 
                obs, reward, done, info = self.envs[env_id].step(responses_str[batch_idx])
                self.env_states[env_id]['step'] += 1
                self.env_states[env_id]['done'] = done
                self.env_states[env_id]['metrics']['traj_metrics'] = info['metrics'].get('traj_metrics', {})
                for k,v in info['metrics']['turn_metrics'].items():
                    self.env_states[env_id]['metrics']['turn_metrics'][k].append(v)
                
                self.record(env_id, obs, reward, done, info)
                # log the output of the rollout
                #print(f"- rollout_loop: env_id={env_id}, step={step}\n- {reward=} \n- action_content={info['action_content']}\n- output_str={responses_str[batch_idx]}")

                #print(f"[Debug] env_id={env_id}, step={step}, obs_str: {obs['obs_str']}")
        
    @torch.no_grad()
    def generate_batch_for_update(self) -> DataProto:
        """
        Get the final trajectory of all environments

        Returns:
            batch (DataProto): batch of final trajectory of all environments
        """
        batch_list = []
        for env_id in self.envs.keys():
            row_dict = self._generate_input_for_uptate(
                recording=self.recorder[env_id],
                step=self.env_states[env_id]['step'],
                window_size=None,
            )
            row_dict['reward_model'] = {"style": "given", "ground_truth": {"reward": self.envs[env_id].compute_reward()}}
            batch_list.append(row_dict)
                    
        # -- Start of 0627 ADD
        # Extra reward to encourage using tool in multi-turn if the model rarely use tool in the batch
        # Store original rewards for comparison (deep copy of tensor values)
        # original_rewards_for_check = []
        # for row_dict in batch_list:
        #     if 'multi_turn_token_level_rewards' in row_dict:
        #         original_rewards_for_check.append(row_dict['multi_turn_token_level_rewards'].clone())
        #     else:
        #         original_rewards_for_check.append(None)
        # 0705 ADD: when using stepwise frame reward, it should also be answer-binded
        
        batch_list = self._get_answer_binded_stepwise_tool_reward(batch_list)


        batch_list = self._compute_batch_tool_usage_reward(batch_list)
        
        # Check if any rewards were modified
        # rewards_modified = False
        # for i, row_dict in enumerate(batch_list):
        #     if 'multi_turn_token_level_rewards' in row_dict and original_rewards_for_check[i] is not None:
        #         if not torch.equal(row_dict['multi_turn_token_level_rewards'], original_rewards_for_check[i]):
        #             rewards_modified = True
        #             break
        
        # if rewards_modified:
        #     print(f"[Debug] batch list rewards were modified")
        # else:
        #     print(f"[Debug] batch list rewards were not modified")
        # -- End of 0627 ADD

        batch_dict = collate_fn(batch_list)

        batch = DataProto.from_single_dict(batch_dict)
        return batch
    
    @torch.no_grad()
    def recording_to_log(self):
        """
        Get the recording of all environments
        
        Returns:
            Dictionary containing the recording of all environments
        """
        env_info = []
        for env_id, record in self.recorder.items():
            config_id = self.envs[env_id].config.config_id()
            step= self.env_states[env_id]['step']
            output_rst = self._single_recording_to_prompt(record, self.env_states[env_id]['step'], window_size=None, is_final=False)
            image= output_rst['image_data']
            done = self.env_states[env_id]['done']
            score = self.envs[env_id].compute_reward()
            
            metrics={
                "score": score,
                "done": done,
                "step": step,
            }
            
            turn_metrics={
                k: sum(v)/step if step != 0 else 0 for k, v in self.env_states[env_id]['metrics']['turn_metrics'].items()
            }
            traj_metrics=self.env_states[env_id]['metrics']['traj_metrics']
            metrics.update(turn_metrics)
            metrics.update(traj_metrics)
            env_info.append({
                "env_id": env_id,
                "config_id": config_id,
                "output_str": output_rst['prompt'],
                "image_data": image,
                "metrics": metrics,
            })
        return env_info
            
            