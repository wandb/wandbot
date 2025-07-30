from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModalConfig(BaseSettings):
    """Configuration for Modal deployment settings"""
    
    model_config = SettingsConfigDict(
        env_prefix="MODAL_", 
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API server settings
    api_cpu: float = Field(4.0, description="CPU cores for API server")
    api_memory: int = Field(4096, description="Memory in MB for API server")
    api_min_containers: int = Field(1, description="Minimum containers to keep warm")
    api_max_containers: int = Field(5, description="Maximum concurrent containers")
    api_max_inputs: int = Field(10, description="Maximum concurrent inputs per container")
    api_timeout: int = Field(600, description="Timeout in seconds for API requests")
    
    # Bot settings
    bot_cpu: float = Field(1.0, description="CPU cores for bots")
    bot_memory: int = Field(2048, description="Memory in MB for bots")
    bot_min_containers: int = Field(1, description="Minimum bot containers to keep warm")
    bot_timeout: int = Field(86400, description="Timeout in seconds for bots (24 hours)")
    
    # Modal app settings
    app_name: str = Field("wandbot-api", description="Modal app name")
    secrets_name: str = Field("wandbot-secrets", description="Modal secrets name")
    
    # Image settings
    python_version: str = Field("3.12", description="Python version for base image")
    base_image: str = Field("debian_slim", description="Base image type")