"""
Nilink Configuration
====================
Centralized settings loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with defaults suitable for development."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Logging
    LOG_LEVEL: str = "info"

    # Rate limiting
    RATE_LIMIT: str = "60/minute"
    RATE_LIMIT_BATCH: str = "10/minute"

    # Batch
    MAX_BATCH_SIZE: int = 10

    # CORS
    CORS_ORIGINS: str = "*"

    # Engine
    MAX_LATENCY_MS: float = 66.0

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
