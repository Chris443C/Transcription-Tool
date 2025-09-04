"""Configuration management system for the interview transcription application."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    default_boost_level: int = 3
    min_boost_level: int = 1
    max_boost_level: int = 10
    output_quality: str = "0"  # FFmpeg audio quality
    codec: str = "libmp3lame"
    supported_formats: tuple = (".mp4", ".mp3", ".wav", ".m4a")


@dataclass
class TranscriptionConfig:
    """Transcription service configuration."""
    model: str = "medium"
    task: str = "translate"
    output_format: str = "srt"
    language: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class TranslationConfig:
    """Translation service configuration."""
    default_source_lang: str = "auto"
    default_target_lang: str = "en"
    # Support for dual language translation
    enable_dual_translation: bool = False
    primary_target_lang: str = "en"
    secondary_target_lang: str = "es"
    supported_languages: Dict[str, str] = field(default_factory=lambda: {
        "Auto Detect": "auto",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar",
        "Hindi": "hi",
        "Ukrainian": "uk",
        "Polish": "pl",
        "Dutch": "nl",
        "Swedish": "sv",
        "Norwegian": "no",
        "Danish": "da",
        "Finnish": "fi"
    })


@dataclass
class ProcessingConfig:
    """Processing engine configuration."""
    max_concurrent_jobs: int = 3
    job_timeout: int = 3600  # seconds
    retry_attempts: int = 2
    temp_dir: Optional[str] = None


@dataclass
class DirectoryConfig:
    """Directory configuration."""
    output_dir: str = "output_subtitles"
    boosted_audio_dir: str = "boosted_audio"
    temp_dir: str = "temp"
    log_dir: str = "logs"

    def __post_init__(self):
        """Ensure all directories exist."""
        for dir_path in [self.output_dir, self.boosted_audio_dir, self.temp_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ConfigurationManager:
    """Configuration management with file persistence."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self._config: Optional[ApplicationConfig] = None
    
    @property
    def config(self) -> ApplicationConfig:
        """Get the current configuration, loading from file if not cached."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def load(self) -> ApplicationConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._dict_to_config(data)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                logger.info("Using default configuration")
        
        # Return default configuration
        config = ApplicationConfig()
        self.save(config)  # Save default config for future reference
        return config
    
    def save(self, config: Optional[ApplicationConfig] = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_to_dict(config), f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        config_dict = self._config_to_dict(self.config)
        self._deep_update(config_dict, kwargs)
        self._config = self._dict_to_config(config_dict)
        self.save()
    
    def _config_to_dict(self, config: ApplicationConfig) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "audio": {
                "default_boost_level": config.audio.default_boost_level,
                "min_boost_level": config.audio.min_boost_level,
                "max_boost_level": config.audio.max_boost_level,
                "output_quality": config.audio.output_quality,
                "codec": config.audio.codec,
            },
            "transcription": {
                "model": config.transcription.model,
                "task": config.transcription.task,
                "output_format": config.transcription.output_format,
                "language": config.transcription.language,
                "device": config.transcription.device,
            },
            "translation": {
                "default_source_lang": config.translation.default_source_lang,
                "default_target_lang": config.translation.default_target_lang,
                "enable_dual_translation": config.translation.enable_dual_translation,
                "primary_target_lang": config.translation.primary_target_lang,
                "secondary_target_lang": config.translation.secondary_target_lang,
            },
            "processing": {
                "max_concurrent_jobs": config.processing.max_concurrent_jobs,
                "job_timeout": config.processing.job_timeout,
                "retry_attempts": config.processing.retry_attempts,
                "temp_dir": config.processing.temp_dir,
            },
            "directories": {
                "output_dir": config.directories.output_dir,
                "boosted_audio_dir": config.directories.boosted_audio_dir,
                "temp_dir": config.directories.temp_dir,
                "log_dir": config.directories.log_dir,
            },
            "log_level": config.log_level,
            "log_format": config.log_format,
        }
    
    def _dict_to_config(self, data: dict) -> ApplicationConfig:
        """Convert dictionary to configuration object."""
        config = ApplicationConfig()
        
        if "audio" in data:
            audio_data = data["audio"]
            config.audio = AudioConfig(
                default_boost_level=audio_data.get("default_boost_level", 3),
                min_boost_level=audio_data.get("min_boost_level", 1),
                max_boost_level=audio_data.get("max_boost_level", 10),
                output_quality=audio_data.get("output_quality", "0"),
                codec=audio_data.get("codec", "libmp3lame"),
            )
        
        if "transcription" in data:
            trans_data = data["transcription"]
            config.transcription = TranscriptionConfig(
                model=trans_data.get("model", "medium"),
                task=trans_data.get("task", "translate"),
                output_format=trans_data.get("output_format", "srt"),
                language=trans_data.get("language"),
                device=trans_data.get("device", "auto"),
            )
        
        if "translation" in data:
            transl_data = data["translation"]
            config.translation = TranslationConfig(
                default_source_lang=transl_data.get("default_source_lang", "auto"),
                default_target_lang=transl_data.get("default_target_lang", "en"),
                enable_dual_translation=transl_data.get("enable_dual_translation", False),
                primary_target_lang=transl_data.get("primary_target_lang", "en"),
                secondary_target_lang=transl_data.get("secondary_target_lang", "es"),
            )
        
        if "processing" in data:
            proc_data = data["processing"]
            config.processing = ProcessingConfig(
                max_concurrent_jobs=proc_data.get("max_concurrent_jobs", 3),
                job_timeout=proc_data.get("job_timeout", 3600),
                retry_attempts=proc_data.get("retry_attempts", 2),
                temp_dir=proc_data.get("temp_dir"),
            )
        
        if "directories" in data:
            dir_data = data["directories"]
            config.directories = DirectoryConfig(
                output_dir=dir_data.get("output_dir", "output_subtitles"),
                boosted_audio_dir=dir_data.get("boosted_audio_dir", "boosted_audio"),
                temp_dir=dir_data.get("temp_dir", "temp"),
                log_dir=dir_data.get("log_dir", "logs"),
            )
        
        config.log_level = data.get("log_level", "INFO")
        config.log_format = data.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        return config
    
    def _deep_update(self, base_dict: dict, update_dict: dict) -> None:
        """Deep update dictionary values."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Global configuration instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_config() -> ApplicationConfig:
    """Get the current application configuration."""
    return get_config_manager().config


def setup_logging(config: Optional[ApplicationConfig] = None) -> None:
    """Setup logging configuration."""
    if config is None:
        config = get_config()
    
    # Create logs directory
    log_dir = Path(config.directories.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.log_format,
        handlers=[
            logging.FileHandler(log_dir / "application.log"),
            logging.StreamHandler()
        ]
    )