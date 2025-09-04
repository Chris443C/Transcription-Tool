"""Batch processing service that orchestrates all media processing operations."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..config.configuration import get_config
from ..core.processing_engine import JobResult, JobType, Job
from .audio_service import AudioService
from .transcription_service import TranscriptionService
from .translation_service import TranslationService

logger = logging.getLogger(__name__)


class BatchProcessingService:
    """Service for orchestrating batch media processing operations."""
    
    def __init__(self):
        self.config = get_config()
        self.audio_service = AudioService()
        self.transcription_service = TranscriptionService()
        self.translation_service = TranslationService()
    
    async def process_batch(
        self,
        input_files: List[str],
        output_dir: str,
        boosted_audio_dir: str,
        boost_audio: bool = True,
        extract_subtitles: bool = True,
        boost_level: int = 3,
        translate_subtitles: bool = False,
        src_lang: str = "auto",
        dest_lang: str = "en",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Process a batch of media files."""
        
        total_files = len(input_files)
        processed_files = []
        failed_files = []
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(boosted_audio_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting batch processing of {total_files} files")
        
        for i, media_file in enumerate(input_files):
            try:
                file_progress_start = i / total_files
                file_progress_end = (i + 1) / total_files
                
                if progress_callback:
                    progress_callback(file_progress_start, f"Processing {Path(media_file).name}")
                
                # Create per-file progress callback
                def file_progress_callback(sub_progress: float, sub_message: str):
                    if progress_callback:
                        overall_progress = file_progress_start + (sub_progress * (file_progress_end - file_progress_start))
                        progress_callback(overall_progress, sub_message)
                
                result = await self.process_single_file(
                    media_file=media_file,
                    output_dir=output_dir,
                    boosted_audio_dir=boosted_audio_dir,
                    boost_audio=boost_audio,
                    extract_subtitles=extract_subtitles,
                    boost_level=boost_level,
                    translate_subtitles=translate_subtitles,
                    src_lang=src_lang,
                    dest_lang=dest_lang,
                    progress_callback=file_progress_callback
                )
                
                processed_files.append(result)
                logger.info(f"Successfully processed {media_file}")
                
            except Exception as e:
                error_result = {
                    "input_file": media_file,
                    "error": str(e),
                    "success": False
                }
                failed_files.append(error_result)
                logger.error(f"Failed to process {media_file}: {e}")
        
        if progress_callback:
            progress_callback(1.0, "Batch processing complete")
        
        result = {
            "total_files": total_files,
            "processed_files": len(processed_files),
            "failed_files": len(failed_files),
            "results": processed_files,
            "failures": failed_files,
            "success": len(failed_files) == 0
        }
        
        logger.info(f"Batch processing complete: {len(processed_files)} successful, {len(failed_files)} failed")
        return result
    
    async def process_single_file(
        self,
        media_file: str,
        output_dir: str,
        boosted_audio_dir: str,
        boost_audio: bool = True,
        extract_subtitles: bool = True,
        boost_level: int = 3,
        translate_subtitles: bool = False,
        src_lang: str = "auto",
        dest_lang: str = "en",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Process a single media file through the complete pipeline."""
        
        media_path = Path(media_file)
        file_name = media_path.stem
        result = {
            "input_file": str(media_path),
            "file_name": file_name,
            "success": True,
            "outputs": {}
        }
        
        try:
            # Step 1: Audio Processing (30% of progress)
            if progress_callback:
                progress_callback(0.0, f"Processing audio for {file_name}")
            
            audio_result = await self.audio_service.process_media_file(
                media_file=str(media_path),
                output_dir=output_dir,
                boosted_audio_dir=boosted_audio_dir,
                boost_audio=boost_audio,
                boost_level=boost_level
            )
            
            result["outputs"]["audio"] = audio_result
            audio_for_processing = audio_result.get("audio_for_processing")
            
            if progress_callback:
                progress_callback(0.3, f"Audio processing complete for {file_name}")
            
            # Step 2: Transcription (60% of progress)
            if extract_subtitles and audio_for_processing:
                if progress_callback:
                    progress_callback(0.3, f"Transcribing {file_name}")
                
                transcription_result = await self.transcription_service.transcribe(
                    audio_path=audio_for_processing,
                    output_dir=output_dir,
                    model=self.config.transcription.model,
                    task=self.config.transcription.task,
                    output_format=self.config.transcription.output_format
                )
                
                result["outputs"]["transcription"] = transcription_result
                
                if progress_callback:
                    progress_callback(0.6, f"Transcription complete for {file_name}")
                
                # Step 3: Translation (100% of progress)
                if translate_subtitles and self.translation_service.is_available():
                    subtitle_file = transcription_result.get("output_file")
                    
                    if subtitle_file:
                        if progress_callback:
                            progress_callback(0.6, f"Translating subtitles for {file_name}")
                        
                        # Create translated subtitle filename
                        translated_subtitle_file = Path(output_dir) / f"{dest_lang.upper()}_{file_name}.srt"
                        
                        def translation_progress(prog, msg):
                            if progress_callback:
                                overall_prog = 0.6 + (prog * 0.4)  # 60% to 100%
                                progress_callback(overall_prog, msg)
                        
                        translation_result = await self.translation_service.translate_subtitles(
                            input_file=subtitle_file,
                            output_file=str(translated_subtitle_file),
                            src_lang=src_lang,
                            dest_lang=dest_lang,
                            progress_callback=translation_progress
                        )
                        
                        result["outputs"]["translation"] = translation_result
            
            if progress_callback:
                progress_callback(1.0, f"Processing complete for {file_name}")
            
            return result
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"Error processing {media_file}: {e}")
            raise


# Job handler for batch processing
async def handle_batch_process_job(job: Job) -> JobResult:
    """Handle batch processing job."""
    try:
        batch_service = BatchProcessingService()
        input_data = job.input_data
        
        # Extract parameters from job data
        input_files = input_data.get("input_files", [])
        output_dir = input_data.get("output_dir")
        boosted_audio_dir = input_data.get("boosted_audio_dir")
        boost_audio = input_data.get("boost_audio", True)
        extract_subtitles = input_data.get("extract_subtitles", True)
        boost_level = input_data.get("boost_level", 3)
        translate_subtitles = input_data.get("translate_subtitles", False)
        src_lang = input_data.get("src_lang", "auto")
        dest_lang = input_data.get("dest_lang", "en")
        
        if not input_files or not output_dir or not boosted_audio_dir:
            return JobResult(success=False, error="Missing required parameters")
        
        # Progress callback for job updates
        async def progress_callback(progress: float, message: str):
            from ..core.processing_engine import get_processing_engine
            engine = await get_processing_engine()
            await engine.update_job_progress(job.id, progress, message)
        
        result = await batch_service.process_batch(
            input_files=input_files,
            output_dir=output_dir,
            boosted_audio_dir=boosted_audio_dir,
            boost_audio=boost_audio,
            extract_subtitles=extract_subtitles,
            boost_level=boost_level,
            translate_subtitles=translate_subtitles,
            src_lang=src_lang,
            dest_lang=dest_lang,
            progress_callback=progress_callback
        )
        
        # Collect all output files
        output_files = []
        for file_result in result.get("results", []):
            outputs = file_result.get("outputs", {})
            
            # Collect audio files
            if "audio" in outputs:
                audio_outputs = outputs["audio"]
                for key, value in audio_outputs.items():
                    if key.endswith("_audio") or key == "audio_for_processing":
                        output_files.append(value)
            
            # Collect transcription files
            if "transcription" in outputs:
                transcription_output = outputs["transcription"].get("output_file")
                if transcription_output:
                    output_files.append(transcription_output)
            
            # Collect translation files
            if "translation" in outputs:
                translation_output = outputs["translation"].get("output_file")
                if translation_output:
                    output_files.append(translation_output)
        
        return JobResult(
            success=result["success"],
            data=result,
            output_files=output_files
        )
        
    except Exception as e:
        logger.error(f"Batch processing job failed: {e}")
        return JobResult(success=False, error=str(e))


# Register job handlers
def register_batch_handlers(processing_engine):
    """Register batch processing job handlers."""
    processing_engine.register_job_handler(JobType.BATCH_PROCESS, handle_batch_process_job)