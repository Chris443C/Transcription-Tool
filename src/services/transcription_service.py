"""Transcription service using OpenAI Whisper."""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os

from ..config.configuration import get_config
from ..core.processing_engine import JobResult, JobType, Job

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Exception raised for transcription errors."""
    pass


class WhisperTranscriptionService:
    """Service for transcribing audio using OpenAI Whisper."""
    
    def __init__(self):
        self.config = get_config()
        self.supported_models = [
            "tiny", "tiny.en",
            "base", "base.en", 
            "small", "small.en",
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3", "large"
        ]
        self.supported_formats = ["srt", "vtt", "txt", "json", "tsv"]
    
    async def transcribe_audio(
        self,
        audio_path: str,
        output_dir: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        output_format: str = "srt"
    ) -> Dict[str, str]:
        """Transcribe audio file using Whisper."""
        
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use config defaults if not specified
        if model is None:
            model = self.config.transcription.model
        if output_format is None:
            output_format = self.config.transcription.output_format
        if task is None:
            task = self.config.transcription.task
        
        # Validate parameters
        if model not in self.supported_models:
            logger.warning(f"Model {model} not in supported list, using anyway")
        
        if output_format not in self.supported_formats:
            raise TranscriptionError(f"Unsupported output format: {output_format}")
        
        if task not in ["transcribe", "translate"]:
            raise TranscriptionError(f"Unsupported task: {task}")
        
        # Build Whisper command
        cmd = [
            "whisper",
            str(audio_path),
            "--model", model,
            "--task", task,
            "--output_format", output_format,
            "--output_dir", str(output_dir)
        ]
        
        # Add language if specified
        if language and language != "auto":
            cmd.extend(["--language", language])
        
        # Add device specification if configured
        if hasattr(self.config.transcription, 'device') and self.config.transcription.device != "auto":
            if self.config.transcription.device == "cpu":
                cmd.extend(["--device", "cpu"])
            elif self.config.transcription.device == "cuda":
                cmd.extend(["--device", "cuda"])
        
        logger.info(f"Starting transcription of {audio_path} with model {model}")
        logger.debug(f"Whisper command: {' '.join(cmd)}")
        
        try:
            # Run Whisper command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown Whisper error"
                raise TranscriptionError(f"Whisper failed: {error_msg}")
            
            # Determine output file path
            file_stem = audio_path.stem
            if output_format == "srt":
                output_file = output_dir / f"{file_stem}.srt"
            elif output_format == "vtt":
                output_file = output_dir / f"{file_stem}.vtt"
            elif output_format == "txt":
                output_file = output_dir / f"{file_stem}.txt"
            elif output_format == "json":
                output_file = output_dir / f"{file_stem}.json"
            elif output_format == "tsv":
                output_file = output_dir / f"{file_stem}.tsv"
            else:
                # Default to srt
                output_file = output_dir / f"{file_stem}.srt"
            
            if not output_file.exists():
                raise TranscriptionError(f"Expected output file not created: {output_file}")
            
            logger.info(f"Transcription completed: {output_file}")
            
            result = {
                "output_file": str(output_file),
                "model": model,
                "task": task,
                "format": output_format
            }
            
            # Add language info if available
            if language:
                result["language"] = language
            
            return result
            
        except FileNotFoundError:
            raise TranscriptionError("Whisper not found. Please ensure OpenAI Whisper is installed.")
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")
    
    async def batch_transcribe(
        self,
        audio_files: List[str],
        output_dir: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        output_format: str = "srt",
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, str]]:
        """Transcribe multiple audio files."""
        
        results = []
        total_files = len(audio_files)
        
        for i, audio_file in enumerate(audio_files):
            try:
                if progress_callback:
                    progress = i / total_files
                    progress_callback(progress, f"Transcribing {Path(audio_file).name}")
                
                result = await self.transcribe_audio(
                    audio_file, output_dir, model, language, task, output_format
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file}: {e}")
                results.append({
                    "input_file": audio_file,
                    "error": str(e),
                    "success": False
                })
        
        if progress_callback:
            progress_callback(1.0, "Transcription complete")
        
        return results
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Whisper models."""
        return self.supported_models.copy()
    
    def validate_model(self, model: str) -> bool:
        """Check if model is supported."""
        return model in self.supported_models
    
    async def estimate_transcription_time(self, audio_path: str, model: str = "medium") -> float:
        """Estimate transcription time based on audio duration and model."""
        try:
            # Get audio duration using ffprobe
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                audio_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                info = json.loads(stdout.decode())
                duration = float(info.get("format", {}).get("duration", 0))
                
                # Rough estimation based on model size
                # These are approximate ratios based on typical performance
                model_ratios = {
                    "tiny": 0.05,
                    "tiny.en": 0.05,
                    "base": 0.1,
                    "base.en": 0.1,
                    "small": 0.2,
                    "small.en": 0.2,
                    "medium": 0.4,
                    "medium.en": 0.4,
                    "large": 0.8,
                    "large-v1": 0.8,
                    "large-v2": 0.8,
                    "large-v3": 0.8
                }
                
                ratio = model_ratios.get(model, 0.4)
                estimated_time = duration * ratio
                
                return estimated_time
            
        except Exception as e:
            logger.warning(f"Could not estimate transcription time: {e}")
        
        # Default fallback estimate
        return 60.0  # 1 minute default


class TranscriptionService:
    """Main transcription service that can support multiple backends."""
    
    def __init__(self):
        self.whisper_service = WhisperTranscriptionService()
        self.config = get_config()
    
    async def transcribe(
        self,
        audio_path: str,
        output_dir: str,
        **kwargs
    ) -> Dict[str, str]:
        """Transcribe audio using the configured backend (currently Whisper)."""
        return await self.whisper_service.transcribe_audio(
            audio_path, output_dir, **kwargs
        )
    
    async def batch_transcribe(
        self,
        audio_files: List[str],
        output_dir: str,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Batch transcribe multiple files."""
        return await self.whisper_service.batch_transcribe(
            audio_files, output_dir, progress_callback=progress_callback, **kwargs
        )


# Job handler for transcription
async def handle_transcription_job(job: Job) -> JobResult:
    """Handle transcription job."""
    try:
        transcription_service = TranscriptionService()
        input_data = job.input_data
        
        audio_path = input_data.get("audio_path")
        output_dir = input_data.get("output_dir")
        model = input_data.get("model")
        language = input_data.get("language")
        task = input_data.get("task", "transcribe")
        output_format = input_data.get("output_format", "srt")
        
        if not audio_path or not output_dir:
            return JobResult(success=False, error="Missing required parameters")
        
        # Progress callback for job updates
        async def progress_callback(progress: float, message: str):
            from ..core.processing_engine import get_processing_engine
            engine = await get_processing_engine()
            await engine.update_job_progress(job.id, progress, message)
        
        result = await transcription_service.transcribe(
            audio_path=audio_path,
            output_dir=output_dir,
            model=model,
            language=language,
            task=task,
            output_format=output_format
        )
        
        return JobResult(
            success=True,
            data=result,
            output_files=[result["output_file"]]
        )
        
    except Exception as e:
        logger.error(f"Transcription job failed: {e}")
        return JobResult(success=False, error=str(e))


# Register job handlers
def register_transcription_handlers(processing_engine):
    """Register transcription job handlers."""
    processing_engine.register_job_handler(JobType.TRANSCRIPTION, handle_transcription_job)