"""Audio processing service with async operations and error handling."""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile

from ..config.configuration import get_config
from ..core.processing_engine import JobResult, JobType, Job
from ..core.error_handling import (
    AudioProcessingError, FileValidationError, ResourceExhaustionError,
    with_processing_retry, ErrorContext, get_error_handler
)

logger = logging.getLogger(__name__)


class AudioService:
    """Service for audio extraction and processing operations."""
    
    def __init__(self):
        self.config = get_config()
    
    async def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video file."""
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise AudioProcessingError(f"Input video file not found: {video_path}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command for audio extraction
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-q:a", self.config.audio.output_quality,
            "-map", "a",
            "-y",  # Overwrite output files
            str(output_path)
        ]
        
        logger.info(f"Extracting audio from {video_path} to {output_path}")
        
        try:
            # Run FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                raise AudioProcessingError(f"FFmpeg failed: {error_msg}")
            
            if not output_path.exists():
                raise AudioProcessingError(f"Output file was not created: {output_path}")
            
            logger.info(f"Audio extracted successfully: {output_path}")
            return str(output_path)
            
        except FileNotFoundError:
            raise AudioProcessingError("FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
        except Exception as e:
            raise AudioProcessingError(f"Audio extraction failed: {str(e)}")
    
    async def boost_audio(self, input_path: str, output_path: str, boost_level: int) -> str:
        """Boost audio volume level."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise AudioProcessingError(f"Input audio file not found: {input_path}")
        
        if boost_level < self.config.audio.min_boost_level or boost_level > self.config.audio.max_boost_level:
            raise AudioProcessingError(
                f"Boost level must be between {self.config.audio.min_boost_level} "
                f"and {self.config.audio.max_boost_level}"
            )
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command for audio boosting
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-af", f"volume={boost_level}.0",
            "-c:a", self.config.audio.codec,
            "-y",  # Overwrite output files
            str(output_path)
        ]
        
        logger.info(f"Boosting audio {input_path} by {boost_level}x to {output_path}")
        
        try:
            # Run FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                raise AudioProcessingError(f"FFmpeg failed: {error_msg}")
            
            if not output_path.exists():
                raise AudioProcessingError(f"Output file was not created: {output_path}")
            
            logger.info(f"Audio boosted successfully: {output_path}")
            return str(output_path)
            
        except FileNotFoundError:
            raise AudioProcessingError("FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
        except Exception as e:
            raise AudioProcessingError(f"Audio boosting failed: {str(e)}")
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in self.config.audio.supported_formats
    
    async def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get audio file information using FFprobe."""
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found: {audio_path}")
        
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(audio_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFprobe error"
                raise AudioProcessingError(f"FFprobe failed: {error_msg}")
            
            import json
            return json.loads(stdout.decode())
            
        except FileNotFoundError:
            raise AudioProcessingError("FFprobe not found. Please ensure FFmpeg is installed and in PATH.")
        except json.JSONDecodeError:
            raise AudioProcessingError("Failed to parse FFprobe output")
        except Exception as e:
            raise AudioProcessingError(f"Failed to get audio info: {str(e)}")
    
    async def process_media_file(
        self,
        media_file: str,
        output_dir: str,
        boosted_audio_dir: str,
        boost_audio: bool = True,
        boost_level: int = 3
    ) -> Dict[str, str]:
        """Process a single media file (extract and boost audio if needed)."""
        media_path = Path(media_file)
        file_name = media_path.stem
        file_ext = media_path.suffix.lower()
        
        result = {}
        
        # Determine paths
        audio_file = None
        boosted_audio_file = None
        
        if file_ext == ".mp4":
            # Extract audio from MP4
            audio_file = Path(boosted_audio_dir) / f"{file_name}.mp3"
            audio_file_str = await self.extract_audio(str(media_path), str(audio_file))
            result["extracted_audio"] = audio_file_str
        elif file_ext in [".mp3", ".wav", ".m4a"]:
            # Use audio file directly
            audio_file = media_path
            result["original_audio"] = str(audio_file)
        else:
            raise AudioProcessingError(f"Unsupported file format: {file_ext}")
        
        # Boost audio if requested
        if boost_audio and audio_file:
            boosted_audio_file = Path(boosted_audio_dir) / f"boosted_{file_name}.mp3"
            boosted_audio_str = await self.boost_audio(
                str(audio_file),
                str(boosted_audio_file),
                boost_level
            )
            result["boosted_audio"] = boosted_audio_str
        
        # Return the audio file to use for further processing
        if boost_audio and boosted_audio_file:
            result["audio_for_processing"] = str(boosted_audio_file)
        elif audio_file:
            result["audio_for_processing"] = str(audio_file)
        
        return result
    
    async def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """Clean up temporary audio files."""
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists() and path.parent.name in ["temp", "tmp"]:
                    path.unlink()
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {e}")


# Job handler for audio processing
async def handle_audio_extraction_job(job: Job) -> JobResult:
    """Handle audio extraction job."""
    try:
        audio_service = AudioService()
        input_data = job.input_data
        
        video_path = input_data.get("video_path")
        output_path = input_data.get("output_path")
        
        if not video_path or not output_path:
            return JobResult(success=False, error="Missing required parameters")
        
        result_path = await audio_service.extract_audio(video_path, output_path)
        
        return JobResult(
            success=True,
            data={"output_path": result_path},
            output_files=[result_path]
        )
        
    except Exception as e:
        logger.error(f"Audio extraction job failed: {e}")
        return JobResult(success=False, error=str(e))


async def handle_audio_boost_job(job: Job) -> JobResult:
    """Handle audio boosting job."""
    try:
        audio_service = AudioService()
        input_data = job.input_data
        
        input_path = input_data.get("input_path")
        output_path = input_data.get("output_path")
        boost_level = input_data.get("boost_level", 3)
        
        if not input_path or not output_path:
            return JobResult(success=False, error="Missing required parameters")
        
        result_path = await audio_service.boost_audio(input_path, output_path, boost_level)
        
        return JobResult(
            success=True,
            data={"output_path": result_path},
            output_files=[result_path]
        )
        
    except Exception as e:
        logger.error(f"Audio boost job failed: {e}")
        return JobResult(success=False, error=str(e))


# Register job handlers
def register_audio_handlers(processing_engine):
    """Register audio processing job handlers."""
    processing_engine.register_job_handler(JobType.AUDIO_EXTRACTION, handle_audio_extraction_job)
    processing_engine.register_job_handler(JobType.AUDIO_BOOST, handle_audio_boost_job)