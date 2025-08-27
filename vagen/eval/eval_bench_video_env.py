#!/usr/bin/env python3
"""
Evaluation script using real training components for consistent LLM responses.
This script directly uses the same ActorRolloutRefWorker and RayWorkerGroup 
as used during training to ensure identical behavior.
"""
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import json
import re
import sys
import numpy as np

# # Add VAGEN to Python path
# vagen_path = "/data/xieyuan/VAGEN"
# if vagen_path not in sys.path:
#     sys.path.append(vagen_path)
import torch
import time
import argparse
import yaml
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict
from omegaconf import OmegaConf, open_dict

from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer



# Record main function start time
main_start_time = time.time()
    
parser = argparse.ArgumentParser(description="Evaluation using real training components")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model checkpoint")
parser.add_argument('--file_name', type=str, required=True, help="Output file identifier")
parser.add_argument('--datasets', type=str, nargs='+', required=True, help="List of datasets to evaluate")
parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs (auto-detected from checkpoint if distributed)")
parser.add_argument('--limit_samples', type=int, default=None, help="Limit number of samples for testing")
parser.add_argument('--prefix', type=str, default="", help="Prefix for the output file")
parser.add_argument('--data_root', type=str, default="", help="Root directory for video datasets")
parser.add_argument('--eval_fresh', action='store_true', default=False, help="Start evaluation fresh, ignoring existing results. If not specified, will resume from existing results if available.")
    
args = parser.parse_args()
    
print("🚀 Starting evaluation")
print(f"📂 Model path: {args.model_path}")
print(f"🏷️  File name: {args.file_name}")
print(f"📊 Datasets: {args.datasets}")
    


model_path = args.model_path
prefix = args.prefix

llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len = 8192,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 32, "video": 1},
)

    
sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[],
)

    
from verl.utils import hf_tokenizer, hf_processor
tokenizer = hf_tokenizer(model_path)
processor = hf_processor(model_path, use_fast=True)
print("✅ Initialized hf_tokenizer and processor")


from vagen.rollout.qwen_rollout.rollout_manager_without_wg import QwenVLRolloutManager_WithoutWG
from vagen.env.video.env import VideoEnv
from vagen.env.video.env_config import VideoEnvConfig
from vagen.env.video.prompt import TYPE_TEMPLATE, parse_videnv_llm_raw_response
from vagen.env.video.reward import reward_fn
from vagen.env import REGISTERED_ENV

def create_video_data_config(video_data_config_path, test_anno_path, dataset_name, data_root=''):
    """Create video data configuration file"""
    # Use the provided data_root parameter

    yaml_data = {
        'dataset': {
            'data_root': data_root,
            'train_anno_path': None,
            'test_anno_path': test_anno_path
        },
        'video_env': {
            'max_video_frames': 64,
            'fps': 1,
            'init_sample_num': 16,
            'max_turns': 3,
            'max_turn_frames': 8
        }
    }
    with open(video_data_config_path, 'w') as f:
        yaml.dump(yaml_data, f)

def extract_multiturn_response_from_raw_str(raw_str):
    turns_response = []
    """Extract answer from multi-turn conversation raw string"""
    try:
        assistant_turns = []
        parts = raw_str.split('<|im_start|>assistant')
        turn_idx = 0
        for part in parts[1:]: 
            turn_idx += 1
            if '<|im_end|>' in part:
                assistant_output = part.split('<|im_end|>')[0].strip()
                if assistant_output:
                    print(f"- turn_{turn_idx}: {assistant_output}")
                    assistant_turns.append(assistant_output)
        
        for turn_output in assistant_turns:
            
            try:
                parsed = parse_videnv_llm_raw_response(turn_output, max_frame_idx=63)
                turns_response.append(parsed.get("llm_response", ""))

            except:
                continue
        
                
        return turns_response
        
    except Exception as e:
        print(f"Error extracting answer from raw_str: {e}")
        return ""

def multi_turn_evaluation_with_training_rollout(rollout_manager, env_configs):
    """Perform multi-turn evaluation using the real training rollout manager"""
    try:
        # Reset environments with the provided configs
        rollout_manager.reset(env_configs)
        
        # Run rollout loop - same as training
        rollout_manager.rollout_loop()
        
        # Get results from recording - same format as training
        results = rollout_manager.recording_to_log()
        
        # Process results
        processed_results = []
        for result in results:
            env_id = result['env_id']
            final_answer = ""
            final_reward = result['metrics'].get('score', 0.0)
            success = result['metrics'].get('success', False)
            
            output_str = result['output_str']
            
            # Extract answer from multi-turn conversation using enhanced function
            multiturn_response = extract_multiturn_response_from_raw_str(output_str)


            
            processed_results.append({
                "env_id": env_id,
                "final_answer": final_answer,
                "reward": final_reward,
                "success": success,
                "metrics": result['metrics'],
                "output_str": output_str
            })
        
        return processed_results
        
    except Exception as e:
        print(f"Error in multi_turn_evaluation_with_training_rollout: {e}")
        return []

def extract_answer(text):
    """Extract answer using VAGEN's parsing function"""
    try:
        parsed = parse_videnv_llm_raw_response(text, max_frame_idx=0)
        if parsed.get("action_type") == "answer":
            return parsed.get("action_content", "")
    except:
        pass
    
    # Fallback to pattern matching
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except:
        return None

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()

def reward_fn_eval(sample, model_output, question_type):
    try:
        output_ans = extract_answer(model_output)
        if output_ans == '':
            output_ans = model_output
        gt_ans = sample.get("solution", "")
        
        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        else:
            return 0.0
    except Exception as e:
        return 0.0

def evaluate_dataset(dataset_name, llm, tokenizer, processor, file_name, prefix="", limit_samples=None, eval_fresh=True, data_root=""):
    """Evaluate a single dataset using training components"""
    
    print(f"\n🔄 Starting evaluation for dataset: {dataset_name}")
    dataset_start_time = time.time()
    
    # Setup paths 
    OUTPUT_DIR = f"./vagen/eval/results/{prefix}" if prefix else "./vagen/eval/results"
    # Create results directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_PATH = f"{OUTPUT_DIR}/eval_{dataset_name}_{file_name}_hftokenizer.json"

    ANNO_PATH = f"./data/eval/eval_{dataset_name}.json"


    
    # Load dataset
    if ANNO_PATH.endswith('.jsonl'):
        data = []
        with open(ANNO_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif ANNO_PATH.endswith('.json'):
        with open(ANNO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")
    
    if limit_samples:
        data = data[:limit_samples]
        print(f"⚠️  Limited to {limit_samples} samples for testing")
    
    print(f"📊 Total samples in {dataset_name}: {len(data)}")


    resume_index = 0
    # Handle checkpoint resume logic
    final_output = []
    if eval_fresh:
        print("🆕 Starting fresh evaluation")
    else:
        # Try to load existing results and resume from checkpoint
        existing_results = load_existing_results(OUTPUT_PATH)
        resume_index, final_output = find_resume_point(data, existing_results)
        
        # If all samples are already processed, return early
        if resume_index >= len(data):
            print(f"✅ Dataset {dataset_name} already completed, skipping")
            # Calculate proper statistics from existing results
            existing_success_count = len([r for r in existing_results if r.get('multiturn_success', False)])
            existing_total = len(existing_results)
            existing_accuracy = 0.0
            if existing_results:
                rewards = [r.get('reward', 0.0) for r in existing_results if 'reward' in r]
                existing_accuracy = sum(rewards) / len(rewards) if rewards else 0.0
            
            return {
                'dataset': dataset_name,
                'time': 0.0,
                'accuracy': existing_accuracy,
                'success_count': existing_success_count,
                'total_processed': existing_total,
                'success_rate': (existing_success_count / existing_total * 100) if existing_total > 0 else 0
            }
        
        # Slice data to start from resume point
        # data = data[resume_index:]
        print(f"🔄 Continuing with {len(data)-resume_index} remaining samples")
    
    # Create video data config
    try:
        create_video_data_config("./vagen/eval/eval_video-data.yaml", ANNO_PATH, dataset_name, data_root)
        print(f"✅ Created video data config for {dataset_name}")
    except Exception as e:
        print(f"❌ Error creating video data config: {e}")
        return None
    
    # Initialize rollout manager with training-compatible components
    config = OmegaConf.create({
        'max_trajectory_length': 8192,
        'truncation': 'left',
        'use_multi_turn_reward': True,
        'use_loss_mask': True,
        'use_gae_mask': True,
        'max_turns': 3,
        'window_size': 2,
        'n_trajectory': 1,
        'mini_batch_size': 64,
        'special_token_for_loss_mask': ['<|box_start|>', '<|box_end|>'],
        'use_service': False,
        'n_gpus_per_node': torch.cuda.device_count()
    })

    rollout_manager = QwenVLRolloutManager_WithoutWG(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        processor=processor,
        sampling_params=sampling_params
    )
    
    print("✅ Initialized QwenVLRolloutManager")
    
    # Process samples
    mean_acc = []
    mean_mra = []
    success_count = 0
    total_processed = 0
    
    # In resume mode, calculate previous statistics from existing results
    if not eval_fresh and resume_index > 0:
        print(f"🔄 Resume mode: Calculating previous statistics from {len(final_output)} existing results")
        for existing_result in final_output:
            if existing_result.get('multiturn_success', False):
                success_count += 1
            total_processed += 1
            
            # Also add to accuracy statistics if available
            if 'reward' in existing_result:
                if existing_result.get('problem_type', '') != 'regression':
                    mean_acc.append(existing_result['reward'])
                else:
                    mean_mra.append(existing_result['reward'])
        
        print(f"📊 Previous statistics: Success: {success_count}/{total_processed} ({success_count/total_processed*100:.2f}%)")
    
    batch_size = 32 #torch.cuda.device_count()
    print(f"📦 Processing in batches of size: {batch_size}")
    
    for batch_start in tqdm(range(0+resume_index, len(data), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(data))
        batch_samples = data[batch_start:batch_end]
        
        try:
            # Create env_configs for the batch
            env_configs = []
            for i, sample in enumerate(batch_samples):
                env_config = {
                    "env_name": "video",
                    "env_config": {
                        'render_mode': 'vision',
                        "video_data_config_path": "./vagen/eval/eval_video-data.yaml"
                    },
                    "seed": batch_start + i,
                    "split": "test"
                }
                env_configs.append(env_config)
            
            # Perform evaluation using REAL training rollout manager
            batch_results = multi_turn_evaluation_with_training_rollout(rollout_manager, env_configs)

            
            # Process results for each sample in the batch
            for i, (sample, result) in enumerate(zip(batch_samples, batch_results)):
                try:
                    computed_reward = reward_fn_eval(sample, result["final_answer"], sample.get("problem_type", ""))
                except:
                    computed_reward = result["reward"]
                
                # Store results
                sample["output"] = "" #result["final_answer"] 
                sample["prediction"] = result["final_answer"]
                sample["reward"] = computed_reward
                sample['correct'] = computed_reward > 0
                sample["turn_history"] = result["output_str"]
                sample["multiturn_success"] = result["success"]
                                
                if result["success"]:
                    success_count += 1
                total_processed += 1
                
                if sample['problem_type'] != 'regression':
                    mean_acc.append(computed_reward)
                else:
                    mean_mra.append(computed_reward)
                    
                final_output.append(sample)
                
        except Exception as e:
            print(f'❌ Error processing batch {batch_start}-{batch_end}: {e}')
            # Add error entries for the batch
            for sample in batch_samples:
                sample["output"] = "error"
                sample["prediction"] = "error"
                sample["reward"] = 0.0
                sample['correct'] = False
                sample["turn_history"] = ""
                sample["multiturn_success"] = False
                total_processed += 1
                final_output.append(sample)

        # Save progress after each batch
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            
            success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
            print(f"✅ Processed batch {batch_start}-{batch_end}. Success: {success_count}/{total_processed} ({success_rate:.2f}%)")
            

            
        except Exception as e:
            print(f"❌ Error writing to output file: {e}")

    # Calculate final metrics
    final_acc = {'mean_acc': 0.0, 'mean_mra': 0.0}
    if mean_acc:
        final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if mean_mra:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    final_acc['success_count'] = success_count
    final_acc['total_processed'] = total_processed
    final_acc['success_rate'] = success_rate
    
    # Save final results
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"💾 Final results saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ Error writing final results: {e}")
    
    dataset_end_time = time.time()
    dataset_time = dataset_end_time - dataset_start_time
    
    print(f"🎯 Dataset {dataset_name} Results:")
    print(f"   ⏱️  Time: {dataset_time:.2f} seconds")
    print(f"   📈 Mean accuracy: {final_acc['mean_acc']:.4f}")
    print(f"   ✅ Success: {success_count}/{total_processed} ({success_rate:.2f}%)")
    
    return {
        'dataset': dataset_name,
        'time': dataset_time,
        'accuracy': final_acc['mean_acc'],
        'success_count': success_count,
        'total_processed': total_processed,
        'success_rate': success_rate
    }

def load_existing_results(output_path):
    """Load existing evaluation results for checkpoint resume"""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_results = existing_data.get('results', [])
                print(f"🔄 Found existing results with {len(existing_results)} samples")
                return existing_results
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}, starting fresh")
    return []

def find_resume_point(data, existing_results):
    """Find the point to resume evaluation from"""
    if not existing_results:
        print("🎯 No existing results, starting from beginning")
        return 0, []
    
    # For most datasets, we can use the length of existing results as the resume point
    # assuming the data order is consistent
    resume_index = len(existing_results)
    
    # Ensure we don't exceed the data length
    if resume_index >= len(data):
        print(f"✅ All {len(data)} samples already processed")
        return len(data), existing_results
    
    print(f"🎯 Resuming from sample {resume_index} (skipping {len(existing_results)} completed samples)")
    return resume_index, existing_results


# Run evaluation on all datasets
total_start_time = time.time()
all_results = []
    
for dataset_name in args.datasets:
    try:
        result = evaluate_dataset(
            dataset_name, 
            llm, tokenizer, processor, args.file_name, prefix,
            limit_samples=args.limit_samples, eval_fresh=args.eval_fresh, data_root=args.data_root
        )
        if result:
            all_results.append(result)
    except Exception as e:
        print(f"❌ Error evaluating dataset {dataset_name}: {e}")
 
    
    # Print summary
    total_time = time.time() - total_start_time
    print(f"\n🏁 EVALUATION COMPLETE")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Results summary:")
    
    total_success = 0
    total_samples = 0
    
    for result in all_results:
        print(f"   {result['dataset']}: {result['success_rate']:.2f}%, {result['success_count']}/{result['total_processed']} ")
        total_success += result['success_count']
        total_samples += result['total_processed']
    
    if total_samples > 0:
        overall_success_rate = (total_success / total_samples * 100)
        print(f"\n🎯 OVERALL: {total_success}/{total_samples} ({overall_success_rate:.2f}%) success across all datasets")
    
    
    # Calculate and output total main function runtime
    main_end_time = time.time()
    main_total_time = main_end_time - main_start_time
    print(f"\n⏱️  TOTAL MAIN FUNCTION RUNTIME: {main_total_time:.2f} seconds ({main_total_time/60:.2f} minutes)")
    print(f"🕐 Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(main_start_time))}")
    print(f"🕐 Ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(main_end_time))}")

