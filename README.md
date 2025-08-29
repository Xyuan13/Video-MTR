# Video-MTR: Multi-Turn Reinforcement Learning for Long-form Video Understanding


## Overview

Video-MTR is a framework for training vision-language models (VLMs) on long video understanding tasks using multi-turn reinforcement learning. 
This repo provides:
- Code for training and evaluation.
- Annotations for the 8K temporally grounded QA dataset for training.
Large model checkpoints are released separately on Hugging Face: https://huggingface.co/Phoebe13/Video-MTR.

## Project Structure

```
Video-MTR/
├── data/                    # Data annotations and intermediate data processing files
├── scripts/                 # Training and evaluation scripts
├── vagen/                   # Core VAGEN framework components
├── qwen-vl-utils/           # Qwen VL model utilities
├── hf.sh                    # Hugging Face cache configuration
└── README.md               # This file
```

## Components

###  `data/`
Contains training data annotations for video understanding tasks. The annotations are in JSON format

**Note**: Please download the original video datasets from their respective sources and place them in the appropriate directories as specified in the annotations.

### `scripts/`
Training  scripts for the Video-MTR framework:

- **`video_ppo/`**: Contains PPO (Proximal Policy Optimization) training scripts
  - `run_7B.sh`: Main training script for 7B parameter models
  - `video-data-qv-nextgqa.yaml`: Data configuration for video Q&A tasks
  - `env_config.yaml`: Environment configuration settings

### `vagen/`
Core VAGEN framework components adapted for video understanding:

- **`eval/`**: Evaluation utilities and metrics
- **`utils/`**: Common utility functions
- **`trainer/`**: Training loop implementations
- **`server/`**: Distributed training server components
- **`rollout/`**: Rollout management for RL training
- **`inference/`**: Model inference utilities

###  `qwen-vl-utils/`
Utilities and tools for working with Qwen VL models.

## Installation

1. **Clone the repository**:
```
# Create a new conda environment
conda create -n vidoe-mtr python=3.10 -y
conda activate vidoe-mtr

# verl
git clone https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

# Video-MTR
git clone https://github.com/Xyuan13/Video-MTR
cd Video-MTR
bash install.sh

# Set up Hugging Face cache** (optional):
```bash
# Edit hf.sh to set your desired cache directory
bash hf.sh
```


## Usage

### Training


**Hardware Requirements:**
- Training the 7B model requires **8 GPUs** 

```bash

# Run training with 7B model
bash scripts/video_ppo/run_7B.sh
```

### Evaluation
**Hardware Requirements:**
- Evaluate the 7B model requires **4 GPUs** 
```bash

bash vagen/eval/run_eval.sh
```

## Acknowledgements

We thank **[VAGEN](https://github.com/RAGEN-AI/VAGEN)** for providing the foundational multi-turn reinforcement learning infrastructure. This code is based on VAGEN with modifications specifically tailored for video understanding tasks. 

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation:

```bibtex
@misc{xie2025videomtrreinforcedmultiturnreasoning,
      title={Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding}, 
      author={Yuan Xie and Tianshui Chen and Zheng Ge and Lionel Ni},
      year={2025},
      eprint={2508.20478},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.20478}, 
}
```
