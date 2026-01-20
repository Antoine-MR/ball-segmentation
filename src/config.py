from pathlib import Path
from typing import Any

import yaml


class Config:
    """Singleton over configs/config.yaml."""
    _instance: "Config | None" = None
    _config: Any = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.reload()

    def reload(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a dotted key, e.g. 'models.sam2_1.path'."""
        value = self._config
        for k in key.split('.'):
            if not isinstance(value, dict):
                return default
            value = value.get(k)
            if value is None:
                return default
        return value

    def get_path(self, key: str) -> Path:
        value = self.get(key)
        if value is None:
            raise ValueError(f"Path configuration '{key}' not found")
        return Path(value)

    @property
    def config(self) -> dict:
        return self._config


config = Config()
