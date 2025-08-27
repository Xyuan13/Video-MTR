# First, import the video environment
from .video import VideoEnv, VideoEnvConfig, VideoEnvService

REGISTERED_ENV = {
    "video": {
        "env_cls": VideoEnv,
        "config_cls": VideoEnvConfig,
        "service_cls": VideoEnvService,
    }
}



