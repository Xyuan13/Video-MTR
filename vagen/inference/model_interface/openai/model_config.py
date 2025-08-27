# vagen/mllm_agent/model_interface/openai/model_config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from vagen.inference.model_interface.base_model_config import BaseModelConfig


# Default API settings
STEPAI_API_CONFIG = {
    "api_base": "https://models-proxy.stepfun-inc.com/v1/",  # 默认使用StepAI的API
    "api_key": "ak-78e9pqrs34j5klm67nop12abcd45fghi78",  
    "model": "gpt-4o", #"gpt-4o",  # 默认使用的模型
}

@dataclass
class OpenAIModelConfig(BaseModelConfig):
    """Configuration for OpenAI API model interface."""
    
    # OpenAI specific parameters
    api_key: Optional[str] = STEPAI_API_CONFIG["api_key"]  #None  # If None, will use environment variable
    organization: Optional[str] = None
    base_url: Optional[str] = STEPAI_API_CONFIG["api_base"] #None  # For custom endpoints
    
    # Model parameters
    model_name: str = "gpt-4o"
    max_retries: int = 3
    timeout: int = 60
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # Provider identifier
    provider: str = "openai"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        return f"OpenAIModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the OpenAI provider."""
        return {
            "description": "OpenAI API for GPT models",
            "supports_multimodal": True,
            "supported_models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo", 
                "gpt-4-vision-preview",
                "gpt-3.5-turbo"
            ],
            "default_model": "gpt-4o"
        }