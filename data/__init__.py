"""
Centralized configuration management.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
from dotenv import dotenv_values


@dataclass
class APIConfig:
    """Configuration for external APIs."""
    entrez_email: str
    entrez_api_key: Optional[str] = None
    max_requests_per_second: int = 3
    retry_attempts: int = 3
    retry_delay: int = 2


@dataclass
class DataConfig:
    """Configuration for data processing."""
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    min_variant_length: int = 50
    max_variant_length: int = 10_000_000
    include_unknown_length: bool = False


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    test_size: float = 0.2
    cv_folds: int = 10
    random_state: int = 42


@dataclass
class Config:
    """Main configuration object."""
    api: APIConfig
    data: DataConfig
    model: ModelConfig
    
    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Config":
        """Load configuration from environment file."""
        env = dotenv_values(env_file)
        
        return cls(
            api=APIConfig(
                entrez_email=env.get("ENTREZ_EMAIL", ""),
                entrez_api_key=env.get("ENTREZ_API_KEY")
            ),
            data=DataConfig(),
            model=ModelConfig()
        )
    