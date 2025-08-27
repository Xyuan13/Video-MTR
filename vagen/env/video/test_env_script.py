from vagen.env.video.env import VideoEnv
from vagen.env.video.env_config import VideoEnvConfig
# Create environment
config = VideoEnvConfig()
env = VideoEnv(config)

# Reset environment
obs, info = env.reset(seed=42)
print("Initial observation:", obs['obs_str'])

# Test step with mock LLM response
mock_llm_response = "Action1, Action2, Action3"
next_obs, reward, done, info = env.step(mock_llm_response)

print("Next observation:", next_obs['obs_str'])
print("Reward:", reward)
print("Done:", done)
print("Action valid:", info['metrics']['action_is_valid'])
print("Action effective:", info['metrics']['action_is_effective'])

# Clean up
env.close()