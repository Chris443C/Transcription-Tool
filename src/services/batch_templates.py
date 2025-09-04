"""
Batch Templates for Processing Configurations
Provides templates for common processing workflows with preset configurations.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Types of processing templates"""
    QUICK_TRANSCRIPTION = "quick_transcription"
    HIGH_QUALITY = "high_quality"
    PODCAST = "podcast"
    LECTURE = "lecture"
    INTERVIEW = "interview"
    MEETING = "meeting"
    MULTILINGUAL = "multilingual"
    CUSTOM = "custom"

class ProcessingMode(Enum):
    """Processing modes"""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"

@dataclass
class AudioSettings:
    """Audio processing settings"""
    boost_enabled: bool = True
    boost_level: float = 2.0
    normalize_audio: bool = True
    noise_reduction: bool = False
    enhance_speech: bool = False
    sample_rate: int = 16000
    channels: int = 1

@dataclass
class TranscriptionSettings:
    """Transcription settings"""
    model: str = "medium"
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # or "translate"
    beam_size: int = 5
    best_of: int = 5
    temperature: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = True
    prepend_punctuations: str = "\"'"¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：")]}、"

@dataclass
class TranslationSettings:
    """Translation settings"""
    enabled: bool = False
    target_languages: List[str] = field(default_factory=list)
    source_language: Optional[str] = None
    preserve_formatting: bool = True
    batch_translate: bool = True

@dataclass
class OutputSettings:
    """Output format settings"""
    srt_enabled: bool = True
    txt_enabled: bool = True
    vtt_enabled: bool = False
    json_enabled: bool = False
    word_level: bool = False
    include_confidence: bool = False
    max_line_width: int = 42
    max_line_count: int = 2

@dataclass
class PerformanceSettings:
    """Performance and optimization settings"""
    use_gpu: bool = True
    quantization: Optional[str] = "fp16"  # int8, fp16, dynamic, None
    streaming_enabled: bool = False
    chunk_size: int = 30
    parallel_jobs: int = 3
    cache_enabled: bool = True

@dataclass
class BatchTemplate:
    """Complete batch processing template"""
    id: str
    name: str
    description: str
    type: TemplateType
    audio_settings: AudioSettings
    transcription_settings: TranscriptionSettings
    translation_settings: TranslationSettings
    output_settings: OutputSettings
    performance_settings: PerformanceSettings
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    is_built_in: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class BatchTemplateManager:
    """Manages batch processing templates"""
    
    BUILTIN_TEMPLATES = {
        "quick_transcription": BatchTemplate(
            id="quick_transcription",
            name="Quick Transcription",
            description="Fast transcription with basic settings for quick results",
            type=TemplateType.QUICK_TRANSCRIPTION,
            audio_settings=AudioSettings(
                boost_enabled=True,
                boost_level=1.5,
                normalize_audio=True
            ),
            transcription_settings=TranscriptionSettings(
                model="base",
                beam_size=1,
                best_of=1,
                word_timestamps=False
            ),
            translation_settings=TranslationSettings(),
            output_settings=OutputSettings(
                txt_enabled=True,
                srt_enabled=True
            ),
            performance_settings=PerformanceSettings(
                quantization="int8",
                parallel_jobs=4
            ),
            tags=["fast", "basic", "quick"],
            is_built_in=True
        ),
        
        "high_quality": BatchTemplate(
            id="high_quality",
            name="High Quality Transcription",
            description="Maximum quality transcription with advanced settings",
            type=TemplateType.HIGH_QUALITY,
            audio_settings=AudioSettings(
                boost_enabled=True,
                boost_level=2.5,
                normalize_audio=True,
                noise_reduction=True,
                enhance_speech=True
            ),
            transcription_settings=TranscriptionSettings(
                model="large",
                beam_size=5,
                best_of=5,
                temperature=[0.0],
                word_timestamps=True,
                condition_on_previous_text=True
            ),
            translation_settings=TranslationSettings(),
            output_settings=OutputSettings(
                srt_enabled=True,
                txt_enabled=True,
                vtt_enabled=True,
                json_enabled=True,
                include_confidence=True
            ),
            performance_settings=PerformanceSettings(
                quantization="fp16",
                streaming_enabled=True,
                parallel_jobs=2
            ),
            tags=["quality", "detailed", "professional"],
            is_built_in=True
        ),
        
        "podcast": BatchTemplate(
            id="podcast",
            name="Podcast Transcription",
            description="Optimized for podcast and long-form audio content",
            type=TemplateType.PODCAST,
            audio_settings=AudioSettings(
                boost_enabled=True,
                boost_level=2.0,
                normalize_audio=True,
                enhance_speech=True
            ),
            transcription_settings=TranscriptionSettings(
                model="medium",
                language="en",
                beam_size=3,
                word_timestamps=True,
                condition_on_previous_text=True,
                initial_prompt="This is a podcast conversation with natural speech patterns."
            ),
            translation_settings=TranslationSettings(),
            output_settings=OutputSettings(
                srt_enabled=True,
                txt_enabled=True,
                max_line_width=60,
                max_line_count=3
            ),
            performance_settings=PerformanceSettings(
                streaming_enabled=True,
                chunk_size=45,
                parallel_jobs=3
            ),
            tags=["podcast", "conversation", "long-form"],
            is_built_in=True
        ),
        
        "lecture": BatchTemplate(
            id="lecture",
            name="Lecture/Educational Content",
            description="Optimized for educational content and lectures",
            type=TemplateType.LECTURE,
            audio_settings=AudioSettings(
                boost_enabled=True,
                boost_level=2.2,
                normalize_audio=True,
                noise_reduction=True
            ),
            transcription_settings=TranscriptionSettings(
                model="medium",
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=True,
                initial_prompt="This is an educational lecture with technical terminology."
            ),
            translation_settings=TranslationSettings(
                enabled=True,
                target_languages=["es", "fr", "de"]
            ),
            output_settings=OutputSettings(
                srt_enabled=True,
                txt_enabled=True,
                vtt_enabled=True,
                include_confidence=True
            ),
            performance_settings=PerformanceSettings(
                streaming_enabled=True,
                parallel_jobs=2
            ),
            tags=["education", "lecture", "academic"],
            is_built_in=True
        ),
        
        "interview": BatchTemplate(
            id="interview",
            name="Interview Transcription",
            description="Optimized for interviews with multiple speakers",
            type=TemplateType.INTERVIEW,
            audio_settings=AudioSettings(
                boost_enabled=True,
                boost_level=2.3,
                normalize_audio=True,
                enhance_speech=True
            ),
            transcription_settings=TranscriptionSettings(
                model="medium",
                beam_size=4,
                word_timestamps=True,
                condition_on_previous_text=False,  # Better for speaker changes
                initial_prompt="This is an interview with questions and answers."
            ),
            translation_settings=TranslationSettings(),
            output_settings=OutputSettings(
                srt_enabled=True,
                txt_enabled=True,
                word_level=True,
                max_line_width=50
            ),
            performance_settings=PerformanceSettings(
                chunk_size=20,  # Shorter chunks for speaker changes
                parallel_jobs=3
            ),
            tags=["interview", "speakers", "dialogue"],
            is_built_in=True
        ),
        
        "meeting": BatchTemplate(
            id="meeting",
            name="Meeting Recording",
            description="Optimized for business meetings and conference calls",
            type=TemplateType.MEETING,
            audio_settings=AudioSettings(
                boost_enabled=True,
                boost_level=2.8,  # Higher boost for poor audio quality
                normalize_audio=True,
                noise_reduction=True,
                enhance_speech=True
            ),
            transcription_settings=TranscriptionSettings(
                model="medium",
                beam_size=3,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt="This is a business meeting with multiple participants."
            ),
            translation_settings=TranslationSettings(),
            output_settings=OutputSettings(
                srt_enabled=True,
                txt_enabled=True,
                max_line_width=60,
                max_line_count=2
            ),
            performance_settings=PerformanceSettings(
                chunk_size=25,
                parallel_jobs=2
            ),
            tags=["meeting", "business", "conference"],
            is_built_in=True
        ),
        
        "multilingual": BatchTemplate(
            id="multilingual",
            name="Multilingual Content",
            description="For content with multiple languages or translation needs",
            type=TemplateType.MULTILINGUAL,
            audio_settings=AudioSettings(
                boost_enabled=True,
                boost_level=2.0,
                normalize_audio=True
            ),
            transcription_settings=TranscriptionSettings(
                model="large",
                language=None,  # Auto-detect
                task="transcribe",
                beam_size=5,
                word_timestamps=True
            ),
            translation_settings=TranslationSettings(
                enabled=True,
                target_languages=["en", "es", "fr", "de", "ja", "zh"],
                batch_translate=True
            ),
            output_settings=OutputSettings(
                srt_enabled=True,
                txt_enabled=True,
                vtt_enabled=True,
                json_enabled=True
            ),
            performance_settings=PerformanceSettings(
                parallel_jobs=1,  # Conservative for complex processing
                streaming_enabled=True
            ),
            tags=["multilingual", "translation", "international"],
            is_built_in=True
        )
    }
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path.home() / ".config" / "transcription_tool"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.templates_file = self.config_dir / "batch_templates.json"
        
        self.templates: Dict[str, BatchTemplate] = {}
        self.load_templates()
        
    def load_templates(self):
        """Load templates from file and merge with built-ins"""
        # Start with built-in templates
        self.templates = self.BUILTIN_TEMPLATES.copy()
        
        # Load custom templates
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for template_data in data.get('templates', []):
                    template = self._dict_to_template(template_data)
                    if template:
                        self.templates[template.id] = template
                        
                logger.info(f"Loaded {len(data.get('templates', []))} custom templates")
                        
            except Exception as e:
                logger.error(f"Failed to load templates: {e}")
                
    def save_templates(self):
        """Save custom templates to file"""
        try:
            # Only save non-built-in templates
            custom_templates = [
                self._template_to_dict(template)
                for template in self.templates.values()
                if not template.is_built_in
            ]
            
            data = {
                'version': '1.0',
                'templates': custom_templates
            }
            
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.info(f"Saved {len(custom_templates)} custom templates")
            
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
            
    def create_template(self, 
                       name: str, 
                       description: str,
                       base_template_id: str = None,
                       **settings) -> str:
        """Create new custom template"""
        
        # Start with base template or default
        if base_template_id and base_template_id in self.templates:
            base_template = self.templates[base_template_id]
            template = BatchTemplate(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                type=TemplateType.CUSTOM,
                audio_settings=base_template.audio_settings,
                transcription_settings=base_template.transcription_settings,
                translation_settings=base_template.translation_settings,
                output_settings=base_template.output_settings,
                performance_settings=base_template.performance_settings,
                tags=["custom"]
            )
        else:
            template = BatchTemplate(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                type=TemplateType.CUSTOM,
                audio_settings=AudioSettings(),
                transcription_settings=TranscriptionSettings(),
                translation_settings=TranslationSettings(),
                output_settings=OutputSettings(),
                performance_settings=PerformanceSettings(),
                tags=["custom"]
            )
        
        # Apply custom settings
        for key, value in settings.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        template.modified_at = datetime.now()
        
        self.templates[template.id] = template
        self.save_templates()
        
        return template.id
        
    def update_template(self, template_id: str, **updates) -> bool:
        """Update existing template"""
        if template_id not in self.templates:
            return False
            
        template = self.templates[template_id]
        
        if template.is_built_in:
            # Create custom copy of built-in template
            new_id = str(uuid.uuid4())
            template = BatchTemplate(
                id=new_id,
                name=f"{template.name} (Custom)",
                description=template.description,
                type=TemplateType.CUSTOM,
                audio_settings=template.audio_settings,
                transcription_settings=template.transcription_settings,
                translation_settings=template.translation_settings,
                output_settings=template.output_settings,
                performance_settings=template.performance_settings,
                tags=template.tags + ["custom"]
            )
            template_id = new_id
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
                
        template.modified_at = datetime.now()
        
        self.templates[template_id] = template
        self.save_templates()
        
        return True
        
    def delete_template(self, template_id: str) -> bool:
        """Delete custom template"""
        if template_id not in self.templates:
            return False
            
        template = self.templates[template_id]
        if template.is_built_in:
            return False  # Cannot delete built-in templates
            
        del self.templates[template_id]
        self.save_templates()
        
        return True
        
    def get_template(self, template_id: str) -> Optional[BatchTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
        
    def get_templates_by_type(self, template_type: TemplateType) -> List[BatchTemplate]:
        """Get templates by type"""
        return [t for t in self.templates.values() if t.type == template_type]
        
    def get_all_templates(self) -> Dict[str, BatchTemplate]:
        """Get all templates"""
        return self.templates.copy()
        
    def get_builtin_templates(self) -> Dict[str, BatchTemplate]:
        """Get only built-in templates"""
        return {k: v for k, v in self.templates.items() if v.is_built_in}
        
    def get_custom_templates(self) -> Dict[str, BatchTemplate]:
        """Get only custom templates"""
        return {k: v for k, v in self.templates.items() if not v.is_built_in}
        
    def search_templates(self, query: str) -> List[BatchTemplate]:
        """Search templates by name, description, or tags"""
        query = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query in template.name.lower() or 
                query in template.description.lower() or
                any(query in tag.lower() for tag in template.tags)):
                results.append(template)
                
        return results
        
    def export_template(self, template_id: str, file_path: Path) -> bool:
        """Export template to file"""
        if template_id not in self.templates:
            return False
            
        try:
            template = self.templates[template_id]
            template_data = self._template_to_dict(template)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, default=str)
                
            logger.info(f"Exported template {template.name} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export template: {e}")
            return False
            
    def import_template(self, file_path: Path) -> Optional[str]:
        """Import template from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
                
            template = self._dict_to_template(template_data)
            if template:
                # Generate new ID to avoid conflicts
                template.id = str(uuid.uuid4())
                template.is_built_in = False
                template.type = TemplateType.CUSTOM
                
                # Add import tag
                if "imported" not in template.tags:
                    template.tags.append("imported")
                    
                self.templates[template.id] = template
                self.save_templates()
                
                logger.info(f"Imported template: {template.name}")
                return template.id
                
        except Exception as e:
            logger.error(f"Failed to import template: {e}")
            
        return None
        
    def duplicate_template(self, template_id: str, new_name: str = None) -> Optional[str]:
        """Duplicate existing template"""
        if template_id not in self.templates:
            return None
            
        original = self.templates[template_id]
        
        # Create copy
        template = BatchTemplate(
            id=str(uuid.uuid4()),
            name=new_name or f"{original.name} (Copy)",
            description=f"Copy of {original.description}",
            type=TemplateType.CUSTOM,
            audio_settings=AudioSettings(**asdict(original.audio_settings)),
            transcription_settings=TranscriptionSettings(**asdict(original.transcription_settings)),
            translation_settings=TranslationSettings(**asdict(original.translation_settings)),
            output_settings=OutputSettings(**asdict(original.output_settings)),
            performance_settings=PerformanceSettings(**asdict(original.performance_settings)),
            tags=original.tags + ["copy", "custom"],
            is_built_in=False
        )
        
        self.templates[template.id] = template
        self.save_templates()
        
        return template.id
        
    def _template_to_dict(self, template: BatchTemplate) -> Dict[str, Any]:
        """Convert template to dictionary for serialization"""
        return {
            'id': template.id,
            'name': template.name,
            'description': template.description,
            'type': template.type.value,
            'audio_settings': asdict(template.audio_settings),
            'transcription_settings': asdict(template.transcription_settings),
            'translation_settings': asdict(template.translation_settings),
            'output_settings': asdict(template.output_settings),
            'performance_settings': asdict(template.performance_settings),
            'created_at': template.created_at.isoformat(),
            'modified_at': template.modified_at.isoformat(),
            'version': template.version,
            'tags': template.tags,
            'is_built_in': template.is_built_in
        }
        
    def _dict_to_template(self, data: Dict[str, Any]) -> Optional[BatchTemplate]:
        """Convert dictionary to template"""
        try:
            return BatchTemplate(
                id=data['id'],
                name=data['name'],
                description=data['description'],
                type=TemplateType(data['type']),
                audio_settings=AudioSettings(**data['audio_settings']),
                transcription_settings=TranscriptionSettings(**data['transcription_settings']),
                translation_settings=TranslationSettings(**data['translation_settings']),
                output_settings=OutputSettings(**data['output_settings']),
                performance_settings=PerformanceSettings(**data['performance_settings']),
                created_at=datetime.fromisoformat(data['created_at']),
                modified_at=datetime.fromisoformat(data['modified_at']),
                version=data.get('version', '1.0'),
                tags=data.get('tags', []),
                is_built_in=data.get('is_built_in', False)
            )
        except Exception as e:
            logger.error(f"Failed to parse template data: {e}")
            return None
            
    def apply_template_to_config(self, template_id: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply template settings to a configuration dictionary"""
        if template_id not in self.templates:
            return base_config
            
        template = self.templates[template_id]
        
        # Merge template settings into config
        config = base_config.copy()
        
        # Audio settings
        config['audio'] = {**config.get('audio', {}), **asdict(template.audio_settings)}
        
        # Transcription settings  
        config['transcription'] = {**config.get('transcription', {}), **asdict(template.transcription_settings)}
        
        # Translation settings
        config['translation'] = {**config.get('translation', {}), **asdict(template.translation_settings)}
        
        # Output settings
        config['output'] = {**config.get('output', {}), **asdict(template.output_settings)}
        
        # Performance settings
        config['performance'] = {**config.get('performance', {}), **asdict(template.performance_settings)}
        
        return config

# Global template manager instance
template_manager = BatchTemplateManager()