from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
from typing import Optional, List
import os
import yaml
import json

@dataclass
class VideoEnvConfig(BaseEnvConfig):
    """Configuration for Video Environment"""
    dataset_name: str = "path/to/dataset"
    data_dir: str = "vagen/env/video/data"
    render_mode: str = "vision"

    # This will be loaded from env_config.yaml
    video_data_config_path: str =  "SCRIPT_DIR/video-config.yaml"

    seed: int = 42
    # Add your environment-specific parameters here
    prompt_format: str = "free_think_v2" 

    format_reward: float = 0.1
    #frame_reward: float = 0.2 # TODO: add frame reward in config file

    
    yaml_data: dict = field(init=False, default_factory=dict)
    _env_config_path: str = field(init=False, default="")

    def __post_init__(self):        
        raw = self.video_data_config_path
        raw = raw.replace('${env:', '$').replace('}', '')
        # Handle environment variables
        raw = os.path.expandvars(raw)
        
        self.video_data_config_path = raw

        if os.path.exists(self.video_data_config_path):
            with open(self.video_data_config_path, 'r') as f:
                self.yaml_data = yaml.safe_load(f)
        else:
            print(f"Warning: video_data_config file not found: {self.video_data_config_path}")
            self.yaml_data = {}


    def config_id(self) -> str:
        """Generate a unique identifier for this configuration"""
        return f"VideoEnvConfig(dataset={self.dataset_name},seed={self.seed})"

    def generate_seeds(self, size: int, split: str = "train") -> List[int]:
        """
        Generate seeds for video environment based on actual video data.
        
        Args:
            size: Number of seeds to generate
            split: Which split to use ("train" or "test")
            
        Returns:
            List of seeds (video indices)
        """
        if not self.yaml_data:
            print("Warning: No yaml_data available, using simple range")
            return list(range(size))
            
        # Determine which annotation file to use
        if split == "train":
            anno_path = self.yaml_data.get('dataset', {}).get('train_anno_path')
        else:
            anno_path = self.yaml_data.get('dataset', {}).get('test_anno_path')
        
        if not anno_path:
            print(f"Warning: No annotation path found for {split} split")
            return list(range(size))
            
        print(f"VideoEnvConfig: Reading {split} data from {anno_path}")
        
        # Load the annotation file
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                video_data = json.load(f)
            
            # Generate seeds based on available video indices
            available_indices = list(range(len(video_data)))
            
            if size > len(available_indices):
                print(f"Warning: Requested size {size} is larger than available videos {len(available_indices)}")
                size = len(available_indices)
            
            # Return the first 'size' indices as seeds
            seeds = available_indices[:size]

            # shuffle the seeds
            import random
            random.shuffle(seeds)

            print(f"VideoEnvConfig: Generated {len(seeds)} seeds for {split} split")
            return seeds
        else:
            print(f"Warning: Annotation file not found: {anno_path}")
            return list(range(size))  # Fallback to simple range

