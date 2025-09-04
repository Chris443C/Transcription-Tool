"""Translation service for subtitle files."""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    GoogleTranslator = None

try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    PYSRT_AVAILABLE = False
    pysrt = None

from ..config.configuration import get_config
from ..core.processing_engine import JobResult, JobType, Job

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Exception raised for translation errors."""
    pass


class TranslationBackend(ABC):
    """Abstract base class for translation backends."""
    
    @abstractmethod
    async def translate_text(self, text: str, src_lang: str, dest_lang: str) -> str:
        """Translate a single text string."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes."""
        pass


class GoogleTranslationBackend(TranslationBackend):
    """Google Translate backend using deep_translator."""
    
    def __init__(self):
        if not DEEP_TRANSLATOR_AVAILABLE:
            raise TranslationError("deep_translator package is required for Google Translate backend")
        
        self.supported_languages = {
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
        }
    
    async def translate_text(self, text: str, src_lang: str, dest_lang: str) -> str:
        """Translate text using Google Translate."""
        if not text.strip():
            return text
        
        try:
            # Run translation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            translator = GoogleTranslator(source=src_lang, target=dest_lang)
            
            translated = await loop.run_in_executor(
                None,
                translator.translate,
                text
            )
            
            return translated or text
            
        except Exception as e:
            logger.warning(f"Translation failed for text '{text[:50]}...': {e}")
            return text  # Return original text if translation fails
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes."""
        return self.supported_languages.copy()


class SubtitleTranslationService:
    """Service for translating subtitle files."""
    
    def __init__(self, backend: Optional[TranslationBackend] = None):
        if backend is None:
            backend = GoogleTranslationBackend()
        
        self.backend = backend
        self.config = get_config()
        
        if not PYSRT_AVAILABLE:
            raise TranslationError("pysrt package is required for subtitle translation")
    
    async def translate_srt_file(
        self,
        input_file: str,
        output_file: str,
        src_lang: str = "auto",
        dest_lang: str = "en",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Translate an SRT subtitle file."""
        
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise TranslationError(f"Input subtitle file not found: {input_path}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load subtitle file
            subs = pysrt.open(str(input_path))
            total_subs = len(subs)
            
            if total_subs == 0:
                raise TranslationError("No subtitles found in file")
            
            logger.info(f"Translating {total_subs} subtitles from {src_lang} to {dest_lang}")
            
            # Translate each subtitle
            translated_count = 0
            for i, sub in enumerate(subs):
                if sub.text.strip():  # Only translate non-empty subtitles
                    try:
                        sub.text = await self.backend.translate_text(
                            sub.text, src_lang, dest_lang
                        )
                        translated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to translate subtitle {i+1}: {e}")
                        # Keep original text if translation fails
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / total_subs
                    progress_callback(progress, f"Translating subtitle {i+1}/{total_subs}")
            
            # Save translated subtitles
            subs.save(str(output_path), encoding='utf-8')
            
            result = {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "src_lang": src_lang,
                "dest_lang": dest_lang,
                "total_subtitles": total_subs,
                "translated_subtitles": translated_count,
                "success": True
            }
            
            logger.info(f"Translation completed: {translated_count}/{total_subs} subtitles translated")
            return result
            
        except Exception as e:
            logger.error(f"Subtitle translation failed: {e}")
            raise TranslationError(f"Failed to translate subtitle file: {str(e)}")
    
    async def translate_srt_to_multiple_languages(
        self,
        input_file: str,
        output_files: List[tuple],  # List of (output_file, dest_lang) tuples
        src_lang: str = "auto",
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Translate an SRT subtitle file to multiple target languages."""
        
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise TranslationError(f"Input subtitle file not found: {input_path}")
        
        if not output_files:
            raise TranslationError("No target languages specified")
        
        try:
            # Load subtitle file once
            subs = pysrt.open(str(input_path))
            total_subs = len(subs)
            
            if total_subs == 0:
                raise TranslationError("No subtitles found in file")
            
            logger.info(f"Translating {total_subs} subtitles from {src_lang} to {len(output_files)} languages")
            
            results = []
            total_languages = len(output_files)
            
            for lang_index, (output_file, dest_lang) in enumerate(output_files):
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create a copy of the subtitles for this language
                lang_subs = pysrt.SubRipFile()
                for sub in subs:
                    lang_subs.append(pysrt.SubRipItem(
                        index=sub.index,
                        start=sub.start,
                        end=sub.end,
                        text=sub.text
                    ))
                
                translated_count = 0
                
                # Translate each subtitle for this language
                for i, sub in enumerate(lang_subs):
                    if sub.text.strip():  # Only translate non-empty subtitles
                        try:
                            sub.text = await self.backend.translate_text(
                                sub.text, src_lang, dest_lang
                            )
                            translated_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to translate subtitle {i+1} to {dest_lang}: {e}")
                            # Keep original text if translation fails
                    
                    # Update progress for this language
                    if progress_callback:
                        # Calculate overall progress across all languages
                        sub_progress = (i + 1) / total_subs
                        overall_progress = (lang_index + sub_progress) / total_languages
                        progress_callback(
                            overall_progress, 
                            f"Translating to {dest_lang}: subtitle {i+1}/{total_subs}"
                        )
                
                # Save translated subtitles for this language
                lang_subs.save(str(output_path), encoding='utf-8')
                
                result = {
                    "input_file": str(input_path),
                    "output_file": str(output_path),
                    "src_lang": src_lang,
                    "dest_lang": dest_lang,
                    "total_subtitles": total_subs,
                    "translated_subtitles": translated_count,
                    "success": True
                }
                results.append(result)
                
                logger.info(f"Translation to {dest_lang} completed: {translated_count}/{total_subs} subtitles translated")
            
            if progress_callback:
                progress_callback(1.0, f"Translation complete for all {total_languages} languages")
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-language subtitle translation failed: {e}")
            raise TranslationError(f"Failed to translate subtitle file to multiple languages: {str(e)}")
    
    async def batch_translate_files(
        self,
        file_pairs: List[tuple],  # List of (input_file, output_file) tuples
        src_lang: str = "auto",
        dest_lang: str = "en",
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Translate multiple subtitle files."""
        
        results = []
        total_files = len(file_pairs)
        
        for i, (input_file, output_file) in enumerate(file_pairs):
            try:
                if progress_callback:
                    file_progress = i / total_files
                    progress_callback(file_progress, f"Translating {Path(input_file).name}")
                
                # Create per-file progress callback
                def file_progress_callback(sub_progress, sub_message):
                    if progress_callback:
                        overall_progress = (i + sub_progress) / total_files
                        progress_callback(overall_progress, sub_message)
                
                result = await self.translate_srt_file(
                    input_file, output_file, src_lang, dest_lang, file_progress_callback
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to translate {input_file}: {e}")
                results.append({
                    "input_file": input_file,
                    "output_file": output_file,
                    "error": str(e),
                    "success": False
                })
        
        if progress_callback:
            progress_callback(1.0, "Translation complete")
        
        return results
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes from the backend."""
        return self.backend.get_supported_languages()
    
    def validate_language_code(self, lang_code: str) -> bool:
        """Check if language code is supported."""
        return lang_code in self.backend.get_supported_languages().values()


class TranslationService:
    """Main translation service."""
    
    def __init__(self):
        try:
            self.subtitle_service = SubtitleTranslationService()
        except TranslationError as e:
            logger.warning(f"Translation service initialization failed: {e}")
            self.subtitle_service = None
    
    async def translate_subtitles(
        self,
        input_file: str,
        output_file: str,
        src_lang: str = "auto",
        dest_lang: str = "en",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Translate subtitle file to single target language."""
        if not self.subtitle_service:
            raise TranslationError("Translation service not available")
        
        return await self.subtitle_service.translate_srt_file(
            input_file, output_file, src_lang, dest_lang, progress_callback
        )
    
    async def translate_subtitles_to_multiple_languages(
        self,
        input_file: str,
        target_languages: List[tuple],  # List of (output_file, dest_lang) tuples
        src_lang: str = "auto",
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Translate subtitle file to multiple target languages."""
        if not self.subtitle_service:
            raise TranslationError("Translation service not available")
        
        return await self.subtitle_service.translate_srt_to_multiple_languages(
            input_file, target_languages, src_lang, progress_callback
        )
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        if self.subtitle_service:
            return self.subtitle_service.get_supported_languages()
        return {}
    
    def is_available(self) -> bool:
        """Check if translation service is available."""
        return self.subtitle_service is not None


# Legacy function for backward compatibility
async def translate_srt(input_file: str, output_file: str, src_lang: str = "auto", dest_lang: str = "en") -> bool:
    """Legacy function for translating SRT files."""
    try:
        service = TranslationService()
        if not service.is_available():
            return False
        
        result = await service.translate_subtitles(input_file, output_file, src_lang, dest_lang)
        return result.get("success", False)
        
    except Exception as e:
        logger.error(f"Legacy translation function failed: {e}")
        return False


# New function for dual language translation
async def translate_srt_dual_language(
    input_file: str, 
    output_files: List[tuple],  # List of (output_file, dest_lang) tuples
    src_lang: str = "auto"
) -> List[bool]:
    """Translate SRT file to multiple languages."""
    try:
        service = TranslationService()
        if not service.is_available():
            return [False] * len(output_files)
        
        results = await service.translate_subtitles_to_multiple_languages(
            input_file, output_files, src_lang
        )
        return [result.get("success", False) for result in results]
        
    except Exception as e:
        logger.error(f"Dual language translation function failed: {e}")
        return [False] * len(output_files)


# Job handler for translation
async def handle_translation_job(job: Job) -> JobResult:
    """Handle translation job."""
    try:
        translation_service = TranslationService()
        
        if not translation_service.is_available():
            return JobResult(success=False, error="Translation service not available")
        
        input_data = job.input_data
        
        input_file = input_data.get("input_file")
        output_file = input_data.get("output_file")
        src_lang = input_data.get("src_lang", "auto")
        dest_lang = input_data.get("dest_lang", "en")
        
        if not input_file or not output_file:
            return JobResult(success=False, error="Missing required parameters")
        
        # Progress callback for job updates
        async def progress_callback(progress: float, message: str):
            from ..core.processing_engine import get_processing_engine
            engine = await get_processing_engine()
            await engine.update_job_progress(job.id, progress, message)
        
        result = await translation_service.translate_subtitles(
            input_file=input_file,
            output_file=output_file,
            src_lang=src_lang,
            dest_lang=dest_lang,
            progress_callback=progress_callback
        )
        
        return JobResult(
            success=True,
            data=result,
            output_files=[result["output_file"]]
        )
        
    except Exception as e:
        logger.error(f"Translation job failed: {e}")
        return JobResult(success=False, error=str(e))


# Job handler for dual language translation
async def handle_dual_translation_job(job: Job) -> JobResult:
    """Handle dual language translation job."""
    try:
        translation_service = TranslationService()
        
        if not translation_service.is_available():
            return JobResult(success=False, error="Translation service not available")
        
        input_data = job.input_data
        
        input_file = input_data.get("input_file")
        target_languages = input_data.get("target_languages", [])  # List of (output_file, dest_lang) tuples
        src_lang = input_data.get("src_lang", "auto")
        
        if not input_file or not target_languages:
            return JobResult(success=False, error="Missing required parameters")
        
        # Progress callback for job updates
        async def progress_callback(progress: float, message: str):
            from ..core.processing_engine import get_processing_engine
            engine = await get_processing_engine()
            await engine.update_job_progress(job.id, progress, message)
        
        results = await translation_service.translate_subtitles_to_multiple_languages(
            input_file=input_file,
            target_languages=target_languages,
            src_lang=src_lang,
            progress_callback=progress_callback
        )
        
        # Extract output files from results
        output_files = [result["output_file"] for result in results if result.get("success")]
        
        return JobResult(
            success=any(result.get("success", False) for result in results),
            data=results,
            output_files=output_files
        )
        
    except Exception as e:
        logger.error(f"Dual language translation job failed: {e}")
        return JobResult(success=False, error=str(e))


# Register job handlers
def register_translation_handlers(processing_engine):
    """Register translation job handlers."""
    from ..core.processing_engine import JobType
    processing_engine.register_job_handler(JobType.TRANSLATION, handle_translation_job)
    # Register new dual language job handler
    if hasattr(JobType, 'DUAL_TRANSLATION'):
        processing_engine.register_job_handler(JobType.DUAL_TRANSLATION, handle_dual_translation_job)