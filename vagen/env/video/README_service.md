# VideoEnvService

A batch processing service for Video environments that enables parallel operations across multiple video environment instances.

## Overview

`VideoEnvService` extends the functionality of the single `VideoEnv` to support batch operations with parallel processing. This is particularly useful for:

- **Reinforcement Learning Training**: Running multiple environments simultaneously for data collection
- **Batch Evaluation**: Evaluating models across multiple video samples in parallel
- **Performance Optimization**: Leveraging multi-core systems for faster processing

## Key Features

### 🚀 **Parallel Processing**
- Uses `ThreadPoolExecutor` for concurrent operations
- Configurable worker threads via `max_workers` parameter
- Automatic error isolation (one environment failure doesn't affect others)

### 🔧 **Batch Operations**
- `create_environments_batch()`: Create multiple environments in parallel
- `reset_batch()`: Reset multiple environments with different seeds/parameters
- `step_batch()`: Execute actions across multiple environments simultaneously
- `compute_reward_batch()`: Calculate rewards for multiple environments
- `get_system_prompts_batch()`: Retrieve system prompts from multiple environments
- `close_batch()`: Clean shutdown of multiple environments

### 🎯 **Video-Specific Features**
- Support for video-specific reset parameters (`extra_info` with split information)
- Handling of LLM raw responses as actions with optional verbose mode
- Batch embedder initialization for similarity-based rewards
- Split size querying for dataset management

## Usage Examples

### Basic Setup

```python
from vagen.env.video.service import VideoEnvService
from vagen.env.base.base_service_config import BaseServiceConfig

# Create service with 4 worker threads
service_config = BaseServiceConfig(max_workers=4)
video_service = VideoEnvService(service_config)
```

### Creating Environments

```python
env_configs = {
    'env_0': {
        'env_name': 'video',
        'env_config': {
            'dataset_name': 'qvhighlight',
            'video_data_config_path': 'path/to/config.yaml',
            'prompt_format': 'free_think',
            'seed': 42
        }
    },
    'env_1': {
        'env_name': 'video',
        'env_config': {
            'dataset_name': 'qvhighlight',
            'video_data_config_path': 'path/to/config.yaml',
            'prompt_format': 'free_think',
            'seed': 123
        }
    }
}

video_service.create_environments_batch(env_configs)
```

### Resetting Environments

```python
# Simple reset with seeds
reset_params = {
    'env_0': 0,
    'env_1': 1
}

# Advanced reset with extra parameters
reset_params = {
    'env_0': {'seed': 0, 'extra_info': {'split': 'train'}},
    'env_1': {'seed': 1, 'extra_info': {'split': 'test'}}
}

reset_results = video_service.reset_batch(reset_params)
```

### Taking Steps

```python
# Simple action format
step_actions = {
    'env_0': 'I need to watch the video. <watch>0-5</watch>',
    'env_1': 'Let me analyze this scene. <watch>0-10</watch>'
}

# Advanced action format with verbose mode
step_actions = {
    'env_0': {
        'llm_raw_response': 'I need to watch the video. <watch>0-5</watch>',
        'verbose': True
    },
    'env_1': {
        'llm_raw_response': 'Let me analyze this scene. <watch>0-10</watch>',
        'verbose': False
    }
}

step_results = video_service.step_batch(step_actions)
```

### Getting Information

```python
env_ids = ['env_0', 'env_1']

# Get system prompts
prompts = video_service.get_system_prompts_batch(env_ids)

# Get dataset split sizes
train_sizes = video_service.get_split_size_batch(env_ids, split="train")
test_sizes = video_service.get_split_size_batch(env_ids, split="test")

# Initialize embedders for similarity-based rewards
embedder_status = video_service.init_embedder_batch(env_ids, 
                                                   service_url="http://localhost:5000")

# Compute total rewards
rewards = video_service.compute_reward_batch(env_ids)
```

### Cleanup

```python
# Close specific environments
video_service.close_batch(['env_0', 'env_1'])

# Close all environments
video_service.close_batch()
```

## Error Handling

The service provides robust error handling:

- **Individual Environment Errors**: If one environment fails, others continue processing
- **Graceful Degradation**: Failed operations return error information instead of crashing
- **Consistent Return Format**: All batch operations return dictionaries with consistent structure

```python
# Example error handling
reset_results = video_service.reset_batch({'env_0': 42, 'invalid_env': 123})

for env_id, (obs, info) in reset_results.items():
    if 'error' in info:
        print(f"Environment {env_id} failed: {info['error']}")
    else:
        print(f"Environment {env_id} reset successfully")
```

## Performance Considerations

### Worker Thread Configuration
- **CPU-bound tasks**: Set `max_workers` to number of CPU cores
- **I/O-bound tasks**: Can use more workers than CPU cores
- **Memory constraints**: Fewer workers if environments consume significant memory

### Memory Management
- Each environment loads video data independently
- Consider memory usage when setting `max_workers`
- Use `close_batch()` to free resources when done

### Video Processing
- Video frames are cached to disk for reuse
- First access to a video may be slower (processing + caching)
- Subsequent accesses benefit from cached frames

## Integration with Training Systems

The service is designed to integrate with distributed training systems:

```python
# Example integration pattern
class VideoTrainingManager:
    def __init__(self, num_envs=8, max_workers=4):
        self.service = VideoEnvService(BaseServiceConfig(max_workers=max_workers))
        self.setup_environments(num_envs)
    
    def collect_batch_data(self, actions):
        return self.service.step_batch(actions)
    
    def reset_environments(self, seeds):
        return self.service.reset_batch(seeds)
```

## Testing

Run the test suite to verify functionality:

```bash
python VAGEN/vagen/env/video/test_service.py
```

Run the example to see full functionality:

```bash
python VAGEN/vagen/env/video/service_example.py
```

## Architecture Comparison

| Feature | VideoEnv (Single) | VideoEnvService (Batch) |
|---------|-------------------|-------------------------|
| Environment Count | 1 | Multiple |
| Processing | Sequential | Parallel |
| Error Isolation | N/A | ✅ |
| Resource Efficiency | Low | High |
| Training Integration | Manual | Built-in |
| Scalability | Limited | Excellent |

## Dependencies

- `concurrent.futures`: For parallel processing
- `vagen.env.video.env`: Core VideoEnv implementation
- `vagen.server.serial`: For observation serialization
- Standard video processing dependencies (PIL, torch, etc.) 