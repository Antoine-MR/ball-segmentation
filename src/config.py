from pathlib import Path
import typing
import yaml
from typing import Any

class Config:
    """Global configuration loader"""
    _instance: typing.Any = None
    _config: typing.Any = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.reload()
    
    def reload(self, config_path: Path | None = None):
        """Load or reload configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """Get path configuration and convert to Path object"""
        value = self.get(key)
        if value is None:
            raise ValueError(f"Path configuration '{key}' not found")
        return Path(value)
    
    @property
    def config(self) -> dict:
        """Get full configuration dictionary"""
        return self._config

config = Config()
