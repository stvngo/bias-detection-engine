"""
Production configuration for Bias Lab API
"""
import os
from typing import Optional

class ProductionConfig:
    """Production configuration settings"""
    
    # API Configuration
    API_TITLE = "Bias Lab API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Media Bias Detection and Analysis API"
    
    # Server Configuration
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    WORKERS = int(os.getenv("WORKERS", 1))
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 1400))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.3))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "warning")
    LOG_FORMAT = "json"
    
    # Security Configuration
    CORS_ORIGINS = [
        "https://your-domain.com",
        "https://www.your-domain.com",
        "https://bias-lab-api-xxxxx-uc.a.run.app",  # Cloud Run URL
        "https://your-project-id.appspot.com",      # App Engine URL
    ]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
    
    # Cache Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
    
    # Health Check
    HEALTH_CHECK_ENDPOINT = "/health"
    
    # Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for production")
        return True
