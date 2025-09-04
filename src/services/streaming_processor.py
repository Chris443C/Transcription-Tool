"""
Streaming Processor for Large Audio/Video Files
Handles large files without loading entirely into memory using chunked processing.
"""

import os
import tempfile
import asyncio
import logging
from typing import Iterator, List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import time
import hashlib

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

try:
    import whisper
except ImportError:
    whisper = None

import pysrt
from datetime import timedelta

logger = logging.getLogger(__name__)

@dataclass
class StreamChunk:
    """Represents a chunk of audio/video stream"""
    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    file_path: Path
    size_bytes: int
    sample_rate: int = 16000
    channels: int = 1
    processed: bool = False
    transcription: Optional[str] = None
    subtitle_entries: List[pysrt.SubRipItem] = field(default_factory=list)

@dataclass 
class StreamingConfig:
    """Configuration for streaming processing"""
    chunk_duration: float = 30.0  # seconds per chunk
    overlap_duration: float = 2.0  # overlap between chunks
    max_chunk_size: int = 50 * 1024 * 1024  # 50MB max chunk size
    sample_rate: int = 16000
    channels: int = 1
    audio_codec: str = "mp3"
    temp_dir: Optional[Path] = None
    keep_chunks: bool = False
    parallel_chunks: int = 3

class StreamingProcessor:
    """Processes large media files using streaming chunks"""
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.temp_dir = self.config.temp_dir or Path(tempfile.gettempdir()) / "transcription_streaming"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[StreamChunk] = []
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback
        
    def _update_progress(self, progress: float, message: str):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(progress, message)
            
    async def get_media_info(self, file_path: Path) -> Dict[str, Any]:
        """Get media file information using ffprobe"""
        if not ffmpeg:
            raise ImportError("ffmpeg-python required for streaming processing")
            
        try:
            probe = ffmpeg.probe(str(file_path))
            
            # Find audio stream
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                raise ValueError("No audio stream found in file")
                
            duration = float(probe['format']['duration'])
            sample_rate = int(audio_stream.get('sample_rate', 44100))
            channels = int(audio_stream.get('channels', 2))
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'codec': audio_stream.get('codec_name'),
                'bitrate': audio_stream.get('bit_rate'),
                'size': int(probe['format']['size'])
            }
            
        except Exception as e:
            logger.error(f"Failed to get media info for {file_path}: {e}")
            raise
            
    def _calculate_chunks(self, duration: float) -> List[Tuple[float, float]]:
        """Calculate chunk time ranges"""
        chunks = []
        current_start = 0.0
        
        while current_start < duration:
            chunk_end = min(current_start + self.config.chunk_duration, duration)
            chunks.append((current_start, chunk_end))
            
            # Move to next chunk with overlap consideration
            current_start = chunk_end - self.config.overlap_duration
            if current_start >= chunk_end:
                break
                
        return chunks
        
    async def _extract_audio_chunk(self, 
                                 source_file: Path, 
                                 start_time: float, 
                                 end_time: float, 
                                 chunk_id: int) -> StreamChunk:
        """Extract audio chunk from source file"""
        duration = end_time - start_time
        chunk_filename = f"chunk_{chunk_id:06d}_{start_time:.2f}_{end_time:.2f}.{self.config.audio_codec}"
        chunk_path = self.temp_dir / chunk_filename
        
        try:
            # Use ffmpeg to extract chunk
            stream = ffmpeg.input(str(source_file), ss=start_time, t=duration)
            stream = ffmpeg.output(
                stream,
                str(chunk_path),
                acodec=self.config.audio_codec,
                ar=self.config.sample_rate,
                ac=self.config.channels,
                f=self.config.audio_codec
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Get chunk size
            chunk_size = chunk_path.stat().st_size
            
            chunk = StreamChunk(
                chunk_id=chunk_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                file_path=chunk_path,
                size_bytes=chunk_size,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to extract chunk {chunk_id}: {e}")
            raise
            
    async def create_chunks(self, source_file: Path) -> List[StreamChunk]:
        """Create audio chunks from source file"""
        media_info = await self.get_media_info(source_file)
        duration = media_info['duration']
        
        self._update_progress(0.0, "Calculating chunks...")
        
        chunk_ranges = self._calculate_chunks(duration)
        total_chunks = len(chunk_ranges)
        
        logger.info(f"Creating {total_chunks} chunks for {duration:.2f}s duration")
        
        chunks = []
        semaphore = asyncio.Semaphore(self.config.parallel_chunks)
        
        async def extract_chunk_with_semaphore(chunk_id, start_time, end_time):
            async with semaphore:
                chunk = await self._extract_audio_chunk(source_file, start_time, end_time, chunk_id)
                progress = (chunk_id + 1) / total_chunks * 0.3  # 30% for chunk creation
                self._update_progress(progress, f"Created chunk {chunk_id + 1}/{total_chunks}")
                return chunk
        
        # Create chunks in parallel
        tasks = [
            extract_chunk_with_semaphore(i, start, end)
            for i, (start, end) in enumerate(chunk_ranges)
        ]
        
        chunks = await asyncio.gather(*tasks)
        self.chunks = sorted(chunks, key=lambda c: c.chunk_id)
        
        return self.chunks
        
    async def process_chunk_transcription(self, chunk: StreamChunk, model) -> StreamChunk:
        """Transcribe a single chunk"""
        try:
            # Transcribe chunk
            result = model.transcribe(
                str(chunk.file_path),
                fp16=hasattr(model, 'fp16') and model.fp16,
                language=None,  # auto-detect
                task='transcribe'
            )
            
            chunk.transcription = result.get('text', '').strip()
            
            # Create subtitle entries with proper timing
            segments = result.get('segments', [])
            for segment in segments:
                start_seconds = chunk.start_time + segment['start']
                end_seconds = chunk.start_time + segment['end']
                
                subtitle = pysrt.SubRipItem(
                    index=0,  # Will be renumbered later
                    start=pysrt.SubRipTime(seconds=start_seconds),
                    end=pysrt.SubRipTime(seconds=end_seconds),
                    text=segment['text'].strip()
                )
                
                chunk.subtitle_entries.append(subtitle)
            
            chunk.processed = True
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to transcribe chunk {chunk.chunk_id}: {e}")
            raise
            
    async def transcribe_chunks(self, model, parallel_limit: int = 2) -> List[StreamChunk]:
        """Transcribe all chunks with parallel processing"""
        if not self.chunks:
            raise ValueError("No chunks available for transcription")
            
        total_chunks = len(self.chunks)
        semaphore = asyncio.Semaphore(parallel_limit)
        processed_chunks = []
        
        async def transcribe_with_semaphore(chunk):
            async with semaphore:
                processed_chunk = await self.process_chunk_transcription(chunk, model)
                progress = 0.3 + (len(processed_chunks) + 1) / total_chunks * 0.6  # 30-90% progress
                self._update_progress(
                    progress, 
                    f"Transcribed chunk {processed_chunk.chunk_id + 1}/{total_chunks}"
                )
                return processed_chunk
        
        # Process chunks in parallel
        tasks = [transcribe_with_semaphore(chunk) for chunk in self.chunks]
        processed_chunks = await asyncio.gather(*tasks)
        
        self.chunks = sorted(processed_chunks, key=lambda c: c.chunk_id)
        return self.chunks
        
    def merge_transcriptions(self) -> Tuple[str, List[pysrt.SubRipItem]]:
        """Merge chunk transcriptions into final result"""
        if not self.chunks:
            raise ValueError("No processed chunks available")
            
        full_text_parts = []
        all_subtitles = []
        subtitle_index = 1
        
        # Handle overlapping chunks by deduplication
        processed_text_segments = set()
        
        for chunk in self.chunks:
            if chunk.transcription:
                # Simple deduplication based on text similarity
                text_hash = hashlib.md5(chunk.transcription.encode()).hexdigest()
                if text_hash not in processed_text_segments:
                    full_text_parts.append(chunk.transcription)
                    processed_text_segments.add(text_hash)
            
            # Add subtitle entries
            for subtitle in chunk.subtitle_entries:
                # Check for overlap with previous subtitles
                overlaps = False
                for existing in all_subtitles[-3:]:  # Check last 3 entries for overlap
                    if (abs(subtitle.start.ordinal - existing.start.ordinal) < 2000 and  # 2 second tolerance
                        subtitle.text.strip() == existing.text.strip()):
                        overlaps = True
                        break
                        
                if not overlaps:
                    subtitle.index = subtitle_index
                    all_subtitles.append(subtitle)
                    subtitle_index += 1
        
        full_text = ' '.join(full_text_parts)
        return full_text, all_subtitles
        
    def cleanup_chunks(self):
        """Remove temporary chunk files"""
        if not self.config.keep_chunks:
            for chunk in self.chunks:
                try:
                    if chunk.file_path.exists():
                        chunk.file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup chunk {chunk.file_path}: {e}")
                    
    async def process_streaming_file(self, 
                                   source_file: Path, 
                                   model,
                                   output_dir: Path) -> Dict[str, Any]:
        """Complete streaming processing pipeline"""
        try:
            self._update_progress(0.0, "Starting streaming processing...")
            
            # Create chunks
            await self.create_chunks(source_file)
            
            # Transcribe chunks
            await self.transcribe_chunks(model)
            
            # Merge results
            self._update_progress(0.9, "Merging transcriptions...")
            full_text, subtitles = self.merge_transcriptions()
            
            # Save results
            base_name = source_file.stem
            
            # Save subtitle file
            srt_path = output_dir / f"{base_name}.srt"
            srt_file = pysrt.SubRipFile(items=subtitles)
            srt_file.save(str(srt_path), encoding='utf-8')
            
            # Save full text
            txt_path = output_dir / f"{base_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # Cleanup
            self.cleanup_chunks()
            
            self._update_progress(1.0, "Streaming processing complete!")
            
            return {
                'success': True,
                'text': full_text,
                'subtitle_file': srt_path,
                'text_file': txt_path,
                'chunks_processed': len(self.chunks),
                'total_duration': sum(chunk.duration for chunk in self.chunks)
            }
            
        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            self.cleanup_chunks()
            raise

class StreamingTranscriptionService:
    """High-level service for streaming transcription"""
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        
    async def should_use_streaming(self, file_path: Path) -> bool:
        """Determine if streaming processing should be used"""
        try:
            file_size = file_path.stat().st_size
            # Use streaming for files larger than 100MB
            if file_size > 100 * 1024 * 1024:
                return True
                
            # Check duration for audio/video files
            if ffmpeg:
                try:
                    probe = ffmpeg.probe(str(file_path))
                    duration = float(probe['format']['duration'])
                    # Use streaming for files longer than 10 minutes
                    return duration > 600
                except:
                    pass
                    
            return False
            
        except Exception as e:
            logger.warning(f"Could not determine if streaming needed for {file_path}: {e}")
            return False
            
    async def transcribe_large_file(self, 
                                  file_path: Path, 
                                  output_dir: Path,
                                  model,
                                  progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """Transcribe large file using streaming processing"""
        
        processor = StreamingProcessor(self.config)
        if progress_callback:
            processor.set_progress_callback(progress_callback)
            
        return await processor.process_streaming_file(file_path, model, output_dir)

# Global instance
streaming_service = StreamingTranscriptionService()