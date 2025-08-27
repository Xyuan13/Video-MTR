import os
import torch
import numpy as np
from vagen.env.base.base_env import BaseEnv
from typing import Dict, Tuple
import random
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .prompt import system_prompt, init_observation_template, action_template, format_prompt, invalid_state_observation_template
from .prompt import TYPE_TEMPLATE
from .prompt import parse_videnv_llm_raw_response
from .prompt import SYSTEM_PROMPT_VIDEO_R1_EVAL, video_r1_format_user_prompt
from .prompt import system_prompt_v2, free_think_format_prompt_v2
from .env_config import VideoEnvConfig


# video utils
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image
# reward
from .reward import reward_fn, cosine_similarity, calculate_list_iou
from pprint import pprint


class VideoReaderEnv():
    def __init__(self, config_dict):
        self.config = config_dict['video_env']
        self.data_root = config_dict['dataset']['data_root']
        self.train_anno_path = config_dict['dataset']['train_anno_path']
        self.test_anno_path = config_dict['dataset']['test_anno_path']

        # Load all data once at initialization
        self.json_file = self._read_json_file()

        # Current split management
        self.current_split = "train"  # default split
        self.split_json_file = self.json_file[self.current_split]

        # video env config
        self.init_sample_num = self.config['init_sample_num']
        self.max_video_frames = self.config['max_video_frames']
        self.max_turn_frames = self.config.get('max_turn_frames', 8)  # Read max_turn_frames from config, default is 8

        self.retrieved_frames = set()

        # Current observation frames (used for _render_current_state)
        self.current_obs_frames = []
        self.current_obs_idxs = []

        # add: watching tool using
        self.tool_using_num = 0

        # load all frames
        self.video_frames = None  

        # cache - read from config
        video_reader_config = config_dict.get('video_reader_env', {})
        self.use_cache = video_reader_config.get('use_cache', True)
        self.image_type = video_reader_config.get('image_type', "tensor") # "pil" or "tensor"


    def set_split(self, split):
        """Set the current split and update split_json_file"""
        if split in self.json_file:
            self.current_split = split
            self.split_json_file = self.json_file[split]
            # print(f"[Debug] VideoReaderEnv: Set split to {split}, {len(self.split_json_file)} videos available")
        else:
            raise ValueError(f"Split '{split}' not found in data. Available splits: {list(self.json_file.keys())}")
    
    def get_split_data(self, split):
        """Get data for a specific split without changing current split"""
        if split in self.json_file:
            return self.json_file[split]
        else:
            raise ValueError(f"Split '{split}' not found in data. Available splits: {list(self.json_file.keys())}")
    
    def get_video_dict(self, video_index, split=None):
        """Get video dictionary for a specific index and split"""
        if split is None:
            split = self.current_split
        
        split_data = self.get_split_data(split)
        if 0 <= video_index < len(split_data):
            return split_data[video_index]
        else:
            raise IndexError(f"Video index {video_index} out of range for split '{split}' (0-{len(split_data)-1})")

    def _load_video_frames(self, video_idx, split=None):
        """Load all frames of video[idx] from the video dataset."""
        # Get video dict from specified split or current split
        if split is None:
            video_dict = self.split_json_file[video_idx]
        else:
            split_data = self.get_split_data(split)
            video_dict = split_data[video_idx]

        # construt message to feed to qwen_utils
        x = video_dict
        if "problem_type" in x and x["problem_type"] == 'multiple choice':
            question = x['problem'] + "Options:\n"
            for op in x["options"]:
                question += op + "\n"
        else:
            if "problem" in x:
                question = x['problem']
            else:
                question = x['query']

        try:
            if "path" in x: # Video-R1-Format
                # Process path: remove "./Evaluation" prefix if present
                if x['path'].startswith('./Evaluation'):
                    relative_path = x['path'][len('./Evaluation'):]
                else:
                    relative_path = x['path'][1:]
            elif "vid" in x: # QVHighLights
                relative_path = f"/videos/{x['vid']}.mp4"
            else:
                raise ValueError(f"Unknown video format in recognize video path: {x}")

            # Use IMAGE_FACTOR defined in qwen_vl_utils
            from qwen_vl_utils.vision_process import IMAGE_FACTOR
            image_factor = IMAGE_FACTOR
            

            # Save the processed video frames to the processed video frames path
            last_dir = os.path.basename(self.data_root)

            path_with_ext = f"./data/processed_video_cache/{self.image_type}/max_frames_{self.max_video_frames}_imagefactor_{image_factor}/{last_dir}" + relative_path
            processed_video_frames_path = os.path.splitext(path_with_ext)[0]

            #print(f"[Debug] VideoREnv: processed_video_frames_path: {processed_video_frames_path}")

            cache_exists = False
            if self.use_cache:
                if self.image_type == "pil":
                    cache_exists = os.path.exists(processed_video_frames_path)
                elif self.image_type == "tensor":
                    cache_exists = os.path.exists(processed_video_frames_path + ".pt")
                    
            if cache_exists:
                # load the video frames from the processed video frames path
                if self.image_type == "pil":
                    try:
                        import glob
                        from PIL import Image
                        image_files = sorted(
                            glob.glob(os.path.join(processed_video_frames_path, '*.png')), 
                            key=lambda x: int(os.path.basename(x).split('.')[0])
                        )

                        if image_files:
                            image_list = [Image.open(f) for f in image_files]
                            self.video_sample_fps = None # NotImplemented
                            #print(f"[Debug] VideoREnv: load from cache at {processed_video_frames_path}, {len(image_list)} frames, size: {image_list[0].size}")
                            return image_list
                        else:
                            raise ValueError(f"No image files found at {processed_video_frames_path}")


                    except Exception as e:
                        raise ValueError(f"Could not load from {self.image_type} cache at {processed_video_frames_path}. Error: {e}. Re-processing video.")
                        
                elif self.image_type == "tensor": # tensor
                    tensor_file_path = processed_video_frames_path + ".pt"
                    try:
                        image_list = torch.load(tensor_file_path)
                        if image_list is None or len(image_list) == 0:
                            print(f"[Warning] VideoREnv: Loaded empty cache from {tensor_file_path}, will re-process video")
                            raise ValueError("Empty cache file")
                        self.video_sample_fps = None # NotImplemented
                        #print(f"[Debug] VideoREnv: load from cache at {tensor_file_path}, {len(image_list)} frames, shape: {image_list[0].shape}")
                        return image_list
                    except Exception as cache_error:
                        print(f"[Warning] VideoREnv: Failed to load from cache {tensor_file_path}: {cache_error}")
                        # Remove corrupted cache file and continue to re-process
                        try:
                            os.remove(tensor_file_path)
                            print(f"[Debug] VideoREnv: Removed corrupted cache file {tensor_file_path}")
                        except:
                            pass
                else:
                    raise ValueError(f"Unknown image type: {self.image_type}")

            # If cache does not exist or was empty/corrupt, process video
            vision_info= {'type': 'video', 
                    'video': self.data_root + relative_path,
                    "max_frames": self.max_video_frames}

            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            self.video_sample_fps = video_sample_fps
            #print(f"[Debug] VideoREnv: new video , video_sample_fps: {self.video_sample_fps}, shape: {video_input.shape}")

        
            # turn tensor to list[PIL.Image.Image]
            image_list = video_input
            if self.image_type == "pil":
                image_list = [to_pil_image(frame) for frame in image_list]

            # save the image list to the processed video frames path
            if self.use_cache:
                try:
                    os.makedirs(processed_video_frames_path, exist_ok=True)
                    if self.image_type == "pil":
                        for i, image in enumerate(image_list):
                            image.save(os.path.join(processed_video_frames_path, f"{i}.png"))
                    elif self.image_type == "tensor":
                        tensor_file_path = processed_video_frames_path + ".pt"
                        torch.save(image_list, tensor_file_path)
                except Exception as e:
                    print(f"[Warning] VideoREnv: Failed to save frames to cache: {e}")

            return image_list
        except Exception as e:
            print(f"[Error] VideoReaderEnv: Error when processing vision info for video_idx={video_idx}: {e}")
            print(f"[Error] VideoReaderEnv: Video path: {self.data_root + relative_path if 'relative_path' in locals() else 'Unknown'}")
            print(f"[Error] VideoReaderEnv: Cache path: {processed_video_frames_path + '.pt' if 'processed_video_frames_path' in locals() else 'Unknown'}")
            import traceback
            traceback.print_exc()
            return []        

    def _read_json_file(self):
        """Load the Video dataset."""
        import json
        json_file = {}
        
        def load_json_or_jsonl(file_path):
            if file_path.endswith('.jsonl'):
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
                return data
            else:
                with open(file_path, 'r') as f:
                    return json.load(f)
        
        # Load train data if path is provided
        if self.train_anno_path is not None and self.train_anno_path.strip():  # Ensure train_anno_path is not an empty string
            # parse the path from the root of the project
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            full_path = os.path.join(project_root, self.train_anno_path)
            
            print(f"[Debug] VideoReaderEnv: full_path for train_anno: {full_path}")  # Add debug information
            train_anno = load_json_or_jsonl(full_path)
            json_file['train'] = train_anno
        else:
            json_file['train'] = []
        
        # Load test data if path is provided
        if self.test_anno_path is not None and self.test_anno_path.strip():  # Ensure test_anno_path is not an empty string
            # parse the path from the root of the project
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            full_path = os.path.join(project_root, self.test_anno_path)
            
            print(f"[Debug] VideoReaderEnv: full_path for test_anno: {full_path}")  # Add debug information
            test_anno = load_json_or_jsonl(full_path)
            json_file['test'] = test_anno
        else:
            json_file['test'] = []
        
        return json_file
    
    def reset_video(self, video_index, split=None):
        """
        Reset the environment to initial state
        1、uniformly sample {init_sample_num} frames from the entire video
        """
        # Set split if provided
        if split is not None:
            self.set_split(split)
        
        # Get video dict
        video_dict = self.get_video_dict(video_index, split)
        #pprint(f"[Debug] VideoREnv.reset_video: video_dict: {video_dict}")
        # load all frames of the video
        self.video_frames = self._load_video_frames(video_index, split)
        
        # Check for empty or insufficient video frames
        if len(self.video_frames) == 0:
            print(f"[Error] VideoREnv.reset_video: No video frames loaded! video_dict: {video_dict}")
            raise ValueError(f"Failed to load video frames for video_index={video_index}, path={video_dict.get('path', 'Unknown')}")
        elif len(self.video_frames) < 20:
            print(f"[Warning] VideoREnv.reset_video: len(self.video_frames) < 20, video_dict: {video_dict}")

        # reset current obs
        self.current_obs_frames = []
        self.current_obs_idxs = []
        self.retrieved_frames = set()

        video_path = video_dict['path']
        
        return video_dict

    def init_obs(self):
        """Initialize the observation"""

        # generate uniformly sampled frames
        sample_num = self.init_sample_num
        obs_idxs = np.linspace(0, len(self.video_frames) - 1, sample_num, dtype=int)
        obs_frames = [self.video_frames[i] for i in obs_idxs]

        # Add the retrieved frames to the set
        self.retrieved_frames.update(obs_idxs)

        # Update current observation frames
        self.current_obs_frames = obs_frames
        self.current_obs_idxs = obs_idxs
        
        return obs_frames, obs_idxs

    def update_obs(self, start_idx, end_idx):
        """Update observation by retrive new frames from the given interval"""
        # TODO: exclude the frames that have been retrieved
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(self.video_frames) - 1) # ensure end_idx is within the range of the video

        #print(f"[Debug] update_obs: original start_idx={start_idx}, end_idx={end_idx}")

        if start_idx > end_idx:
            assert False, "Erro in video retrieve : start_idx > end_idx, with start_idx={}, end_idx={}".format(start_idx, end_idx)
            
        # Ensure sample_num is at least 1
        frame_range = end_idx - start_idx + 1  # +1 because we want to include both start and end
        sample_num = min(frame_range, self.max_turn_frames)
        sample_num = max(1, sample_num)  # Ensure at least 1
        
        #print(f"[Debug] update_obs: valide start_idx={start_idx}, end_idx={end_idx}, frame_range={frame_range}, sample_num={sample_num}")
        
        obs_idxs = np.linspace(start_idx, end_idx, sample_num, dtype=int)
        obs_frames = [self.video_frames[i] for i in obs_idxs]

        # Add the retrieved frames to the set
        self.retrieved_frames.update(obs_idxs)

        # Update current observation frames
        self.current_obs_frames = obs_frames
        self.current_obs_idxs = obs_idxs
        
        #print(f"[Debug] update_obs: set current_obs_frames to {len(obs_frames)} frames, idxs: {obs_idxs}")

        return obs_frames, obs_idxs

class VideoEnv(BaseEnv):
    """
    Video Environment for training and evaluating vison language models as agents.
    
    It is designed specifically for Large Language Models (LLMs) as agents, providing visual observations and handling action to retrieve frames.
    """
    def __init__(self, config: VideoEnvConfig):
        super().__init__()
        self.config = config
        self.env = VideoReaderEnv(self.config.yaml_data) 
                    
        # Initialize the underlying environment
        self.total_reward = 0
 
        self.turn_num = 1 
        self.tool_using_num = 0  # Initialize tool_using_num
        self.video_sample_fps = None  # Initialize video_sample_fps

        # Store the format prompt function for later use
        self.format_prompt_func = format_prompt[self.config.prompt_format]

        # Initialize max_turns from config
        self.max_turns = self.config.yaml_data.get('video_env', {}).get('max_turns', 3)

        # Initialize random number generator
        self.rng = random.Random()
        if hasattr(self.config, "seed") and self.config.seed is not None:
            self.rng.seed(self.config.seed)

        # Initialize the video index


        # Initialize the frame reward - read from config
        rewards_config = self.config.yaml_data.get('video_env', {}).get('rewards', {})
        self.add_step_frame_reward = rewards_config.get('add_step_frame_reward', False)
        self.add_final_frame_reward = rewards_config.get('add_final_frame_reward', False)
        self.frame_reward = rewards_config.get('frame_reward', 0.5)

        self.cal_frame_iou = True # 
        self.cal_embeddings = False #True


        if self.cal_embeddings:
            self.init_embedder()
            self.similarity_function = cosine_similarity


    

    def init_embedder(self, service_url="http://localhost:5000"):
        """Initialize ImageBind client instead of loading model directly."""
        try:
            from vagen.env.video.imagebind_service.imagebind_client import ImageBindClient
            self.embedder_client = ImageBindClient(service_url)
        except Exception as e:
            print(f"Failed to initialize ImageBind client: {e}")
            raise
    
    def get_obs_embedding(self, obs):
        """Get observation embeddings using the service."""
        try:
            embeddings = self.embedder_client.encode_video_frames(obs)
            return embeddings
        except Exception as e:
            print(f"Error getting observation embeddings: {e}")
            raise
    
    def get_query_embedding(self, query):
        """Get query embedding using the service."""
        try:
            embedding = self.embedder_client.encode_string_query(query)
            return embedding
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            raise

    def _get_obs_query_avg_similarity(self, turn_num):
        """Get the similarity between obs and query"""
        
        obs_embedding = self.obs_embedding[turn_num]
        query_embedding = self.query_embedding
        similarity = self.similarity_function(obs_embedding, query_embedding)
              
        # Get the average of frames_similarity
        avg_similarity = np.mean(similarity)               
        return avg_similarity
    
    def _get_obs_iou(self):
        """Get the iou of the retrieved frames and the relevant frames"""
        retrieved_frames = self.env.current_obs_idxs
        relevant_frames = self.curr_video_dict["relevant_sample_frames"]
        iou = calculate_list_iou(retrieved_frames, relevant_frames)
       
        return iou
    
    def get_split_size(self, split):
        """Get the number of videos in a specific split"""
        try:
            split_data = self.env.get_split_data(split)
            return len(split_data)
        except ValueError:
            return 0

    def get_frame_reward_by_cross_modal_similarity(self, verbose=False):
        """
        Calculate frame reward based on cross-modal similarity between observation and query.
        
        This function compares the current observation's similarity to the query with
        the previous turn's similarity and provides rewards/penalties accordingly.
        
        Args:
            start_idx: Start frame index for the current retrieve action
            end_idx: End frame index for the current retrieve action
            
        Returns:
            float: Frame reward (positive for improvement, negative for degradation, 0 otherwise)
        """
        # Find the most recent valid similarity value
        prev_turn = self.turn_num - 1
        while prev_turn >= 0 and prev_turn not in self.obs_query_similarity:
            prev_turn -= 1
        
        epsilon = 0.04  # Error tolerance for similarity comparison
        
        # Consider error_epsilon
        if prev_turn >= 0:
            current_similarity = self.obs_query_similarity[self.turn_num]
            prev_similarity = self.obs_query_similarity[prev_turn]
            
            # Reward for improved similarity (considering epsilon tolerance)
            if current_similarity - epsilon > prev_similarity:
                if verbose: 
                    print(f"[Debug] VideoEnv.get_frame_reward_by_cross_modal_similarity: Good Retrieve: "
                       f"(curr_turn: {self.turn_num}, prev_turn: {prev_turn}, "
                       f"curr_sim: {current_similarity:.4f}, prev_sim: {prev_similarity:.4f})")
                return self.frame_reward
            
            # Penalty for degraded similarity (considering epsilon tolerance)
            elif current_similarity + epsilon < prev_similarity:
                if verbose:
                    print(f"[Debug] VideoEnv.get_frame_reward_by_cross_modal_similarity: Bad Retrieve: "
                       f"(curr_turn: {self.turn_num}, prev_turn: {prev_turn}): "
                       f"curr_sim: {current_similarity:.4f}, prev_sim: {prev_similarity:.4f})")
                return -self.frame_reward
        
        # No reward if frame reward is disabled, no previous turn, or similarity is within epsilon range
        return 0.0
     
    def _get_final_frame_reward_by_relevant_frames_iou(self, turn_num, verbose=True):
        """
        Calculate frame reward by the iou of the retrieved frames and the relevant frames
        """
        
        if not self.obs_iou:
            return 0.0
        
        init_iou = self.obs_iou[0]

        # Find the maximum turn_num in the obs_iou dict
        max_turn_num = max(self.obs_iou.keys())
        last_iou = self.obs_iou[max_turn_num]
        
        # Calculate the improvement from initial IoU to final IoU
        epsilon = 0.001
        if last_iou > init_iou + epsilon:
            if verbose:
                print(f"[Debug] VideoEnv.get_final_frame_reward_iou: Good Retrieve: "
                       f"obs_iou: {init_iou:.2f}->{last_iou:.2f}")
            return self.frame_reward
        # hard pos neg: # 0629: remove "elif last_iou < init_iou - epsilon: " because its easy to be hack by LLM by always retrieve the same frames
        #else:
        elif last_iou < init_iou - epsilon:
            if verbose:
                print(f"[Debug] VideoEnv.get_final_frame_reward_iou: Bad Retrieve: "
                       f"obs_iou: {init_iou:.2f}->{last_iou:.2f}")
            return -self.frame_reward #0.0 #
        return 0.0
        
    
    def _get_frame_reward_by_relevant_frames_iou(self, turn_num, verbose=True):
        """
        Calculate frame reward by the iou of the retrieved frames and the relevant frames
        """
        # Check if "relevant_sample_frames" is in the curr_video_dict
        if "relevant_sample_frames" not in self.curr_video_dict:
            print(f"[Debug] VideoEnv.get_frame_reward_iou: relevant_sample_frames not in curr_video_dict")
            return 0.0
        if turn_num == 0:
            # if verbose:
            #     print(f"[Debug] VideoEnv.get_frame_reward_iou: : No Retrieve")
            return 0.0
        
        # Find the most recent valid similarity value
        prev_turn = turn_num - 1
        
        while prev_turn >= 0 and prev_turn not in self.obs_iou:
            prev_turn -= 1
        
        curr_iou = self.obs_iou[turn_num]
        prev_iou = self.obs_iou[prev_turn]

        epsilon = 0.001
        if curr_iou > prev_iou + epsilon:
            if verbose:
                print(f"[Debug] VideoEnv.get_frame_reward_iou: Good Retrieve: "
                       f"turn: {prev_turn}->{turn_num}:  "
                       f" {prev_iou:.2f}->{curr_iou:.2f}")
            return self.frame_reward
        elif curr_iou < prev_iou - epsilon:
            if verbose:
                print(f"[Debug] VideoEnv.get_frame_reward_iou: Bad Retrieve: "
                       f"turn: {prev_turn}->{turn_num}:  "
                       f"{prev_iou:.2f}->{curr_iou:.2f}")
            return 0.0 #-self.frame_reward
        return 0.0

        if verbose:
            print(f"[Debug] VideoEnv.get_frame_reward_by_relevant_frames_iou: \n"
                  f"- iou: {iou:.4f}, \n- retrieved_frames: {retrieved_frames}, \n- relevant_frames: {relevant_frames}")
        
        return iou
    

 
    def step(self, llm_raw_response, verbose=False) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment based on the agent's action.
        
        This method:
        1. Parses the raw LLM response to extract actions
        2. Executes each valid action in sequence
        3. Calculates rewards and metrics
        4. Generates the next observation
        
        The action string is expected to be the raw output from an LLM, which 
        may contain special tokens for thought processes or other structured content.
        
        Args:
            action_str (str): Raw string from LLM containing actions
        
        Returns:
            Tuple[Dict, float, bool, Dict]:
                - obs: Dictionary with observation string and optional image data
                - reward: Numeric reward for the step
                - done: Boolean indicating if episode is complete
                - info: Dictionary containing metrics and parsed action data
        """

        """Process an action from the LLM and return the next state"""
        # undate turn_num : valid or invalid action will be counted as one turn
        self.turn_num += 1

        # Add: pass the current dict to the parse_videnv_llm_raw_response
        rst = parse_videnv_llm_raw_response(llm_raw_response, max_frame_idx=len(self.env.video_frames) - 1, video_dict=self.curr_video_dict)



        self.reward = 0
        self.done = False  # Update based on task completion
        obs = None # Will be set below

        metrics = {
            "turn_metrics": {
                "action_valid": rst['action_valid'], # Need to check
            },
            "traj_metrics": {
                "success": False, # TODO: add success metric
            }
        }

        # Execute valid actions
        if metrics["turn_metrics"]["action_valid"]:                       
            # Add format reward if actions were valid and format is correct
            if rst.get("format_correct", True):
                self.reward += self.config.format_reward  # TODO: add format reward in config file

            # Update environment state based on action
            if rst['action_type'] == 'retrieve':

                action_content = rst['action_content'] # eg. 4,10
                start_idx, end_idx = action_content.split(',')

                obs = self._render(init_obs=False, start_idx=int(start_idx), end_idx=int(end_idx))              
                self.tool_using_num += 1


                if self.add_step_frame_reward :
                    if self.cal_frame_iou:
                        iou_reward = self._get_frame_reward_by_relevant_frames_iou(turn_num=self.turn_num, verbose=True)
                        self.reward += iou_reward
                    else:
                        frame_reward = self.get_frame_reward_by_cross_modal_similarity(verbose=True)
                        self.reward += frame_reward
                    #self.reward += frame_reward


            elif rst['action_type'] == 'answer':
                self.done = True


                # Add: pass the step id to the reward function
                answer_reward = reward_fn(self.curr_video_dict, rst['action_content'], self.curr_video_dict['problem_type'], step_id=self.turn_num, tool_using_num=self.tool_using_num)
                #print(f"[Debug] VideoEnv.step: answer reward: {answer_reward:.2f}")

                # undate metrics
                metrics["traj_metrics"]["success"] = answer_reward > 0


                # get extra frame reward if the answer is correct
                if self.add_final_frame_reward and answer_reward > 0:
                    if self.cal_frame_iou:
                        iou_reward = self._get_final_frame_reward_by_relevant_frames_iou(self.turn_num, verbose=True)
                        self.reward += iou_reward

                self.reward += answer_reward    
                
                # Render the state for answer action
                obs = self._render_state_for_invalid_and_answer_action(is_invalid=False, is_answer=True)
        # Invalid action
        # Add the invalid action message to the observation
        else: # invalid action
            # render current state again
            #obs = self._render_current_state()
            obs = self._render_state_for_invalid_and_answer_action(is_invalid=True, is_answer=False)
            wrong_msg = "Your response in the previous turn is invalid: " + rst["error_message"] + "Please try again and response in correct format."
            obs["obs_str"] = wrong_msg + "\n" + obs["obs_str"]

        # Make sure obs is not None
        assert obs is not None, "obs is None"

        # Accumulate total_reward
        self.total_reward += self.reward
        
        # Update info dict
        llm_response = rst['llm_response']
        info = {
            "metrics": metrics,
            "total_reward": self.total_reward,
            "format_correct": rst['format_correct'],
            "llm_raw_response": llm_raw_response,
            "action_type": rst['action_type'],
            "action_content": rst['action_content'],
            "current_obs_idxs": self.env.current_obs_idxs,
            "retrieved_frames": self.env.retrieved_frames,
            "turn_num": self.turn_num,
        }
        
        if verbose:
            print(f"[Debug] VideoEnv.step: Turn_{self.turn_num}: ")
            #print(f"- think_content: {parsed_results['think_content']}")
            print(f"- action_content: {rst['action_content']}")
            print(f"- action_type: {rst['action_type']}")
            print(f"- format_correct: {rst['format_correct']}")
            print(f"- reward: {self.reward}")
            print(f"- total_reward: {self.total_reward}")
            print(f"- done: {self.done}")
            print(f"- info: {info}")

        return obs, self.reward, self.done, info
    

    def reset(self, seed=None, extra_info=None):
        """Reset the environment with an seed
        read new video[idx] from the dataset
        
        Args:
            seed: Random seed for reproducibility
            extra_info: Optional dictionary containing split and other metadata
            
        Returns:
            Tuple[Dict, Dict]: 
                - obs: Dictionary containing observation string and optional image data
                - info: Empty dictionary for initial state
        """
        #print(f"[Debug] VideoEnv.reset: {seed=}")
        
        # Determine which split to use
        assert extra_info is not None, "extra_info can not be None"
        split = extra_info["split"]
        
        # Get data from VideoReaderEnv
        try:
            data = self.env.get_split_data(split)
            if len(data) > 0:
                video_index = seed % len(data)

            else:
                raise ValueError(f"No {split} data available")
                
        except Exception as e:
            print(f"Error loading video data in VideoEnv.reset: {e}")
            raise
        
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.turn_num = 0  
        self.tool_using_num = 0  # Reset tool_using_num

        # Reset the video frames and get video_dict
        video_dict = self.env.reset_video(video_index, split)
        self.curr_video_dict = video_dict
        
        # Get video_sample_fps from VideoReaderEnv
        self.video_sample_fps = None#self.env.video_sample_fps

        
        # Reset the obs embedding
        # key: turn_num, value: obs_embedding
        self.obs_embedding = {}

        # key: turn_num, value: similarity
        self.obs_query_similarity = {}

        if self.cal_frame_iou:
            self.obs_iou = {}

        # Get query embedding
        query = video_dict['problem']
        if self.cal_embeddings:
            self.query_embedding = self.get_query_embedding(query)

        return self._render(init_obs=True), {}

    def _render(self, init_obs=False, start_idx=None, end_idx=None):
        """
        Render the video environment as an observation.
        
        It formats the observation string
        based on whether this is the initial observation or a subsequent one.
        
        Args:
            init_obs (bool): If True, create initial observation; otherwise create a
                            step observation that includes action results
        
        Returns:
            Dict: Observation dictionary containing:
                - "obs_str": String observation for the LLM
                - "multi_modal_data": Optional dictionary with image data for vision mode
        """
        multi_modal_data = None

        img_placeholder = self.config.image_placeholder
        
        max_frame_idx = len(self.env.video_frames) - 1 if len(self.env.video_frames) > 0 else 0
        
        # Get problem type and options
        problem_type = self.curr_video_dict.get('problem_type', 'unknown') if self.curr_video_dict else 'unknown'
        options = self.curr_video_dict.get('options', [])
      
        format_prompt_text = self.format_prompt_func(
            add_example=False,  # Set to False for init and subsequent observations
            max_frame_idx=max_frame_idx,
            problem_type=problem_type,
            turn_num=self.turn_num+1,  # Next turn's observation, turn_num+1
            max_turns=self.max_turns  # Use max_turns stored during initialization
        )
        # Format the template
        if init_obs:
            obs_frames, obs_idxs  = self.env.init_obs()
       
        else:
            # Subsequent observations include action results
            obs_frames, obs_idxs  = self.env.update_obs(start_idx, end_idx)
        
        problem = self.curr_video_dict['problem']
        observation = img_placeholder # Use the image placeholder in the observation if frame is available

        obs_str = init_observation_template(
                problem=problem,
                problem_type=problem_type,
                options=options,
                observation=observation,
                frame_idx_list=obs_idxs,
                max_frame_idx=max_frame_idx,
                turn_num=self.turn_num+1,  # Next turn's observation, turn_num+1
                max_turns=self.max_turns  # Use max_turns stored during initialization
        ) + "\n" + format_prompt_text   
                
        multi_modal_data = {
                img_placeholder: obs_frames #[convert_numpy_to_PIL(self.gym_env._render_gui(mode='rgb_array'))]
        }

        if self.cal_embeddings:
            # update obs embedding
            new_obs_embedding = self.get_obs_embedding(obs_frames)
            self.obs_embedding[self.turn_num] = new_obs_embedding

            # update the similarity between obs and query
            self.obs_query_similarity[self.turn_num] = self._get_obs_query_avg_similarity(self.turn_num)
            # Format the output
            formatted_similarity = {k: f"{v:.2f}" for k, v in self.obs_query_similarity.items()}

            if start_idx is not None and end_idx is not None:
                print(f"[Debug] VideoEnv._render update sim[{start_idx}, {end_idx}]: turn_{self.turn_num}, {formatted_similarity}")

        if self.cal_frame_iou:
            if "relevant_sample_frames" in self.curr_video_dict.keys():
                self.obs_iou[self.turn_num] = self._get_obs_iou()
                formatted_iou = {k: f"{v:.2f}" for k, v in self.obs_iou.items()}
                # if start_idx is not None and end_idx is not None:
                #     print(f"[Debug] VideoEnv._render update iou: turn_{self.turn_num}, {formatted_iou}")


        # Return observation dictionary with appropriate fields
        return {
                "obs_str": obs_str,
                "obs_idxs": obs_idxs,
                "multi_modal_data": multi_modal_data,
        }

    def system_prompt(self) -> str:
        """
        Get the system prompt for the environment.
        
        Returns a prompt explaining the environment to the LLM agent,
        with different prompts for text and vision modes.
        
        Returns:
            str: System prompt string with environment description and instructions
        """

        return system_prompt_v2().format(max_turns=self.max_turns)

    
    def compute_reward(self) -> float:
        """Calculate final episode reward"""
        return self.total_reward  # Calculate based on task completion
        
    def close(self):
        """Clean up any resources"""
        pass

    def _render_current_state(self):
        """
        Render the current state using current observation frames.
        Used when game ends or action is invalid.
        
        Returns:
            Dict: Observation dictionary with current state
        """
        img_placeholder = self.config.image_placeholder
        
        # Get maximum frame index of the video
        max_frame_idx = len(self.env.video_frames) - 1 if self.env.video_frames else 0
        
        # Get problem type
        problem_type = self.curr_video_dict.get('problem_type', 'unknown') if self.curr_video_dict else 'unknown'
        options = self.curr_video_dict.get('options', [])

        
        format_prompt_text = self.format_prompt_func(
            add_example=False,  # Set to False for current state observations
            max_frame_idx=max_frame_idx,
            problem_type=problem_type,
            turn_num=self.turn_num+1,  # Next turn's observation, turn_num+1
            max_turns=self.max_turns  # Use max_turns stored during initialization
        )
        
        
        if hasattr(self.env, 'current_obs_frames') and self.env.current_obs_frames:
            obs_frames = self.env.current_obs_frames
            obs_idxs = self.env.current_obs_idxs
        else:
            # Fallback to initial frames if no current frames available
            assert False, "No current frames available"
        
        problem = self.curr_video_dict['problem']
        observation = img_placeholder # Use the image placeholder in the observation
        
        obs_str = init_observation_template(
            problem=problem,
            problem_type=problem_type,
            options=options,
            observation=observation,
            frame_idx_list=obs_idxs,
            max_frame_idx=max_frame_idx,
            turn_num=self.turn_num+1,  # Next turn's observation, turn_num+1
            max_turns=self.max_turns  # Use max_turns stored during initialization
        ) + "\n" + format_prompt_text   
                
        multi_modal_data = {
            img_placeholder: obs_frames
        }

        # update obs embedding by copy the last turn's obs_embedding
        if self.cal_embeddings:
            self.obs_embedding[self.turn_num] = self.obs_embedding[self.turn_num-1]
        # if self.cal_frame_iou:
        #     self.obs_iou[self.turn_num] = self.obs_iou[self.turn_num-1]


        return {
            "obs_str": obs_str,
            "obs_idxs": obs_idxs,
            "multi_modal_data": multi_modal_data,
        }

    def _render_state_for_invalid_and_answer_action(self, is_invalid=False, is_answer=False):
        """
        Render the state for invalid and answer action.
        
        Returns:
            Dict: Observation dictionary with current state
        """
        img_placeholder = self.config.image_placeholder
        
        # Get maximum frame index of the video
        max_frame_idx = len(self.env.video_frames) - 1 if len(self.env.video_frames) > 0 else 0
        
        # Get problem type
        problem_type = self.curr_video_dict.get('problem_type', 'unknown') if self.curr_video_dict else 'unknown'
        options = self.curr_video_dict.get('options', [])

        
        format_prompt_text = self.format_prompt_func(
            add_example=False,  # Set to False for current state observations
            max_frame_idx=max_frame_idx,
            problem_type=problem_type,
            turn_num=self.turn_num+1,  # Next turn's observation, turn_num+1
            max_turns=self.max_turns  # Use max_turns stored during initialization
        )            

        obs_frames = None
        obs_idxs = None
        
        problem = self.curr_video_dict['problem']
        observation = img_placeholder # Use the image placeholder in the observation
        
        # obs_str = init_observation_template(
        #     problem=problem,
        #     problem_type=problem_type,
        #     options=options,
        #     observation=observation,
        #     frame_idx_list=obs_idxs,
        #     max_frame_idx=max_frame_idx,
        #     turn_num=self.turn_num+1,  # Next turn's observation, turn_num+1
        #     max_turns=self.max_turns  # Use max_turns stored during initialization
        # ) + "\n" + format_prompt_text
        if is_invalid:
            obs_str = invalid_state_observation_template(
                problem=problem,
                problem_type=problem_type,
                options=options,
                max_frame_idx=max_frame_idx,
                turn_num=self.turn_num+1,  # Next turn's observation, turn_num+1
                max_turns=self.max_turns  # Use max_turns stored during initialization
            ) + "\n" + format_prompt_text    
        elif is_answer:
            obs_str = "You have provided your answer. Turn Ends."

        # no multi_modal_data for invalid and answer action
        # multi_modal_data = {
        #     img_placeholder: None
        # }

        # update obs embedding by copy the last turn's obs_embedding
        # if self.cal_embeddings:
        #     self.obs_embedding[self.turn_num] = self.obs_embedding[self.turn_num-1]
        # if self.cal_frame_iou:
        #     self.obs_iou[self.turn_num] = self.obs_iou[self.turn_num-1]

        return {
            "obs_str": obs_str,
            "obs_idxs": obs_idxs,
            #"multi_modal_data": multi_modal_data,
        }


def save_frames(obs, prefix, save_dir):
    from torchvision.transforms import ToPILImage
    import torch

    images = obs["multi_modal_data"][config.image_placeholder]
    for idx in range(len(images)):
        img = images[idx]
        obs_idx = obs["obs_idxs"][idx]
        
        if isinstance(img, torch.Tensor):
          if img.dtype == torch.float32:
            if img.max() > 1.1:
                img = img.clamp(0, 255) / 255.0  # Normalize to range 0~1
          img = ToPILImage()(img)
        #print(f"saving image for obs_idx: {obs_idx}")
        img.save(f"{save_dir}/{prefix}_{obs_idx}.png")

def get_env_config_from_df(df_path):
    """
    Get env_config from df
    """
    import os
    import pandas as pd

    if os.path.exists(df_path):
        df = pd.read_parquet(df_path)
        print(f'data shape: {df.shape}')
        print(f'columns: {list(df.columns)}')
        print(f'first 5 rows:')
        print(df.head())
    else:
        print('file not found, please check the path')


    # Extract all env_configs
    env_configs = []
    for i, row in df.iterrows():
        extra_info = row['extra_info']
        if isinstance(extra_info, dict) and 'env_config' in extra_info:
            env_configs.append(extra_info)

    print(f'Get {len(env_configs)} env_config')

    return env_configs
