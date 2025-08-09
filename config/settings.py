"""
Configuration settings for the Bias Detection Engine
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Hugging Face Token
    hf_token: Optional[str] = None
    
    # Performance Settings
    max_concurrent_requests: int = 10
    response_timeout_ms: int = 200
    
    # Model Configuration
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    
    # Cache Settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()