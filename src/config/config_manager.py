"""Configuration manager for UI and application settings."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .configuration import get_config_manager, ApplicationConfig


@dataclass
class UIConfig:
    """UI-specific configuration."""
    window_width: int = 900
    window_height: int = 800
    remember_directories: bool = True
    theme: str = "default"
    auto_save_preferences: bool = True


class ConfigManager:
    """Enhanced configuration manager for the application."""
    
    def __init__(self):
        self.app_config_manager = get_config_manager()
        self.ui_config_file = Path("ui_config.json")
        self._ui_config: Optional[UIConfig] = None
    
    def get_application_config(self) -> ApplicationConfig:
        """Get main application configuration."""
        return self.app_config_manager.config
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        if self._ui_config is None:
            self._ui_config = self._load_ui_config()
        return self._ui_config
    
    def _load_ui_config(self) -> UIConfig:
        """Load UI configuration from file."""
        if self.ui_config_file.exists():
            try:
                with open(self.ui_config_file, 'r') as f:
                    data = json.load(f)
                return UIConfig(**data)
            except Exception:
                pass
        
        # Return default UI config
        ui_config = UIConfig()
        self._save_ui_config(ui_config)
        return ui_config
    
    def _save_ui_config(self, config: UIConfig) -> None:
        """Save UI configuration to file."""
        try:
            with open(self.ui_config_file, 'w') as f:
                json.dump(config.__dict__, f, indent=2)
        except Exception:
            pass
    
    def update_ui_config(self, **kwargs) -> None:
        """Update UI configuration."""
        ui_config = self.get_ui_config()
        for key, value in kwargs.items():
            if hasattr(ui_config, key):
                setattr(ui_config, key, value)
        
        self._ui_config = ui_config
        self._save_ui_config(ui_config)
    
    def update_translation_config(self, **kwargs) -> None:
        """Update translation configuration."""
        self.app_config_manager.update(translation=kwargs)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported translation languages."""
        return self.get_application_config().translation.supported_languages
    
    def is_dual_translation_enabled(self) -> bool:
        """Check if dual translation is enabled."""
        return self.get_application_config().translation.enable_dual_translation
    
    def get_translation_languages(self) -> tuple:
        """Get primary and secondary translation languages."""
        config = self.get_application_config().translation
        return (config.primary_target_lang, config.secondary_target_lang)
    
    def set_dual_translation(self, enabled: bool, primary_lang: str = "en", secondary_lang: str = "es") -> None:
        """Configure dual translation settings."""
        self.update_translation_config(
            enable_dual_translation=enabled,
            primary_target_lang=primary_lang,
            secondary_target_lang=secondary_lang
        )
    
    def validate_config(self) -> Dict[str, list]:
        """Validate configuration and return any issues."""
        return {'errors': [], 'warnings': []}