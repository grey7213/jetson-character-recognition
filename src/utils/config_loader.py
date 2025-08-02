"""
Configuration loader utility for the Jetson Character Recognition system.
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Utility class for loading and managing configuration files."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to config directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            Dictionary containing configuration data
        """
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self._configs[config_name] = config
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        return self.load_config("model_config")
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Load camera configuration."""
        return self.load_config("camera_config")
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """
        Update configuration values in memory.
        
        Args:
            config_name: Name of the configuration
            updates: Dictionary of updates to apply
        """
        if config_name not in self._configs:
            self.load_config(config_name)
        
        self._deep_update(self._configs[config_name], updates)
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """
        Save configuration to file.
        
        Args:
            config_name: Name of the configuration file
            config_data: Configuration data to save
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_data, file, default_flow_style=False, indent=2)
        
        self._configs[config_name] = config_data
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigLoader._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Global config loader instance
config_loader = ConfigLoader()
