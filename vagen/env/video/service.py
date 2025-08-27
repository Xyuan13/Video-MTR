from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from vagen.server.serial import serialize_observation

from .env import VideoEnv
from .env_config import VideoEnvConfig
from ..base.base_service_config import BaseServiceConfig

class VideoEnvService(BaseService):
    """
    Service class for Video environments.
    Implements batch operations with parallel processing for efficiency.
    """
    
    def __init__(self, config: BaseServiceConfig):
        """
        Initialize the VideoEnvService.
        
        Args:
            config: Service configuration containing max_workers and other settings
        """
        self.max_workers = config.get('max_workers', 10)
        self.environments = {}
        self.env_configs = {}
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple Video environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
                Each config should contain:
                - env_name: Should be "video"
                - env_config: Video specific configuration
        """
        # Define worker function
        def create_single_env(env_id, config):
            # Verify environment type
            env_name = config.get('env_name', 'video')
            if env_name != 'video':
                return env_id, None, f"Expected environment type 'video', got '{env_name}'"
            
            try:
                # Get Video specific configuration
                env_config_dict = config.get('env_config', {})
                
                # Create environment config
                env_config = VideoEnvConfig(**env_config_dict)
                
                # Create environment
                env = VideoEnv(env_config)
                
                return env_id, (env, env_config), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel creation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all environment creation tasks
            futures = {
                executor.submit(create_single_env, env_id, config): env_id 
                for env_id, config in ids2configs.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error creating environment {env_id}: {error}")
                    continue
                
                env, env_config = result
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple Video environments in parallel.
        
        Args:
            ids2seeds: A dictionary where each key is an environment ID and the corresponding
                     value is a seed value (or None for using default seeding behavior).
                     For VideoEnv, this can also include extra_info parameter.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}
        
        # Define worker function
        def reset_single_env(env_id, reset_params):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                
                # Handle different reset parameter formats
                if isinstance(reset_params, dict):
                    seed = reset_params.get('seed', None)
                    extra_info = reset_params.get('extra_info', None)
                    observation, info = env.reset(seed=seed, extra_info=extra_info)
                else:
                    # Backward compatibility: treat as seed only
                    observation, info = env.reset(seed=reset_params)
                
                # Debug: Check observation before serialization
                print(f"[DEBUG] VideoEnvService - Before serialization for env {env_id}:")
                print(f"  observation type: {type(observation)}")
                if isinstance(observation, dict):
                    for key, value in observation.items():
                        print(f"    {key}: {type(value)}")
                        if key == "obs_idxs" and hasattr(value, 'dtype'):
                            print(f"      obs_idxs is numpy array: shape={value.shape}, dtype={value.dtype}")
                
                serialized_observation = serialize_observation(observation)
                
                # Debug: Check observation after serialization
                print(f"[DEBUG] VideoEnvService - After serialization for env {env_id}:")
                print(f"  serialized_observation type: {type(serialized_observation)}")
                if isinstance(serialized_observation, dict):
                    for key, value in serialized_observation.items():
                        print(f"    {key}: {type(value)}")
                        if key == "obs_idxs" and hasattr(value, 'dtype'):
                            print(f"      obs_idxs STILL numpy array: shape={value.shape}, dtype={value.dtype}")
                
                return env_id, (serialized_observation, info), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel reset
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all reset tasks
            futures = {
                executor.submit(reset_single_env, env_id, seed): env_id 
                for env_id, seed in ids2seeds.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error resetting environment {env_id}: {error}")
                    results[env_id] = ({}, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple Video environments in parallel.
        
        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                       value is the action to execute in that environment.
                       For VideoEnv, action is typically an LLM raw response string.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, reward, done, info)
        """
        results = {}
        
        # Define worker function
        def step_single_env(env_id, action):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                
                # Handle different action parameter formats
                if isinstance(action, dict):
                    llm_raw_response = action.get('llm_raw_response', action.get('action', ''))
                    verbose = action.get('verbose', False)
                    observation, reward, done, info = env.step(llm_raw_response, verbose=verbose)
                else:
                    # Backward compatibility: treat as llm_raw_response string
                    observation, reward, done, info = env.step(action)
                
                # Debug: Check step result before serialization
                print(f"[DEBUG] VideoEnvService step - Before serialization for env {env_id}:")
                print(f"  observation type: {type(observation)}")
                if isinstance(observation, dict):
                    for key, value in observation.items():
                        print(f"    {key}: {type(value)}")
                        if key == "obs_idxs" and hasattr(value, 'dtype'):
                            print(f"      obs_idxs is numpy array: shape={value.shape}, dtype={value.dtype}")
                
                # Use serialize_step_result to handle the entire step result
                from vagen.server.serial import serialize_step_result
                serialized_result = serialize_step_result((observation, reward, done, info))
                
                # Debug: Check step result after serialization
                print(f"[DEBUG] VideoEnvService step - After serialization for env {env_id}:")
                serialized_obs, serialized_reward, serialized_done, serialized_info = serialized_result
                print(f"  serialized_observation type: {type(serialized_obs)}")
                if isinstance(serialized_obs, dict):
                    for key, value in serialized_obs.items():
                        print(f"    {key}: {type(value)}")
                        if key == "obs_idxs" and hasattr(value, 'dtype'):
                            print(f"      obs_idxs STILL numpy array: shape={value.shape}, dtype={value.dtype}")
                
                return env_id, serialized_result, None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel step
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all step tasks
            futures = {
                executor.submit(step_single_env, env_id, action): env_id 
                for env_id, action in ids2actions.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error stepping environment {env_id}: {error}")
                    results[env_id] = ({}, 0.0, True, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """
        Compute the total reward for multiple Video environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its computed total reward
        """
        results = {}
        
        # Define worker function
        def compute_reward_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return env_id, env.compute_reward(), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel computation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all computation tasks
            futures = {
                executor.submit(compute_reward_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error computing reward for environment {env_id}: {error}")
                    results[env_id] = 0.0
                else:
                    results[env_id] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """
        Get system prompts for multiple Video environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its corresponding system prompt string
        """
        results = {}
        
        # Define worker function
        def get_system_prompt_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return env_id, env.system_prompt(), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_system_prompt_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error getting system prompt for environment {env_id}: {error}")
                    results[env_id] = ""
                else:
                    results[env_id] = result
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple Video environments and clean up resources in parallel.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all environments
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        # Define worker function
        def close_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                env.close()
                return None
            except Exception as e:
                return str(e)
        
        # Use ThreadPoolExecutor for parallel closing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all closing tasks
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                error = future.result()
                if error:
                    print(f"Error closing environment: {error}")
        
        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
    
    