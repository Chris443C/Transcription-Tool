"""
Enhanced parallel processing system for the transcription application.
Provides efficient multi-file and pipeline parallelization with resource management.
"""

import logging
import asyncio
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import queue

from ..config.config_manager import ConfigManager


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    """Enhanced job with parallel processing support"""
    job_id: str
    file_path: str
    boost_level: int
    target_languages: List[str]
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    output_files: List[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    worker_id: Optional[int] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []


@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing"""
    max_concurrent_files: int = 2
    enable_pipeline_parallel: bool = True
    share_whisper_models: bool = True
    max_memory_usage_mb: int = 4096
    io_thread_pool_size: int = 4
    cpu_thread_pool_size: int = 2


class ResourceManager:
    """Manages shared resources for parallel processing"""
    
    def __init__(self, config: ParallelProcessingConfig):
        self.config = config
        self._whisper_models = {}
        self._model_locks = {}
        self._memory_usage = 0
        self._active_jobs = 0
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def get_whisper_model(self, model_name: str):
        """Get shared Whisper model instance"""
        with self._lock:
            if model_name not in self._whisper_models:
                # Load model (this would be imported from whisper service)
                self.logger.info(f"Loading Whisper model: {model_name}")
                # Placeholder for actual model loading
                self._whisper_models[model_name] = f"model_{model_name}"
                self._model_locks[model_name] = threading.Lock()
            
            return self._whisper_models[model_name], self._model_locks[model_name]
    
    def acquire_processing_slot(self) -> bool:
        """Acquire a processing slot if available"""
        with self._lock:
            if self._active_jobs < self.config.max_concurrent_files:
                self._active_jobs += 1
                return True
            return False
    
    def release_processing_slot(self):
        """Release a processing slot"""
        with self._lock:
            if self._active_jobs > 0:
                self._active_jobs -= 1
    
    def check_memory_available(self, estimated_mb: int) -> bool:
        """Check if enough memory is available"""
        with self._lock:
            return (self._memory_usage + estimated_mb) <= self.config.max_memory_usage_mb
    
    def allocate_memory(self, mb: int):
        """Allocate memory tracking"""
        with self._lock:
            self._memory_usage += mb
    
    def deallocate_memory(self, mb: int):
        """Deallocate memory tracking"""
        with self._lock:
            self._memory_usage = max(0, self._memory_usage - mb)


class ParallelProcessingFacade:
    """
    Enhanced processing facade with parallel processing capabilities.
    Provides efficient batch processing with resource management.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None,
                 progress_callback: Optional[Callable] = None,
                 parallel_config: Optional[ParallelProcessingConfig] = None):
        
        self.config_manager = config_manager or ConfigManager()
        self.progress_callback = progress_callback
        self.parallel_config = parallel_config or ParallelProcessingConfig()
        
        # Resource management
        self.resource_manager = ResourceManager(self.parallel_config)
        
        # Thread pools for different types of operations
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.parallel_config.io_thread_pool_size,
            thread_name_prefix="IO"
        )
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.parallel_config.cpu_thread_pool_size,
            thread_name_prefix="CPU"
        )
        
        # Job management
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.is_cancelled = False
        
        # Progress tracking
        self._progress_lock = threading.Lock()
        self._total_jobs = 0
        self._completed_jobs = 0
        
        # Services (will be injected or created)
        self.audio_service = None
        self.transcription_service = None
        self.translation_service = None
        
        self.logger = logging.getLogger(__name__)
    
    def set_services(self, audio_service, transcription_service, translation_service):
        """Inject service dependencies"""
        self.audio_service = audio_service
        self.transcription_service = transcription_service
        self.translation_service = translation_service
    
    def process_files_parallel(self, file_paths: List[str], options) -> List[ProcessingJob]:
        """
        Process multiple files with parallel processing.
        Returns list of ProcessingJob objects for tracking progress.
        """
        self.is_cancelled = False
        jobs = []
        
        # Create jobs
        for i, file_path in enumerate(file_paths):
            job = ProcessingJob(
                job_id=f"job_{int(time.time() * 1000)}_{i}",
                file_path=file_path,
                boost_level=options.boost_level,
                target_languages=options.target_languages.copy()
            )
            jobs.append(job)
            self.active_jobs[job.job_id] = job
        
        self._total_jobs = len(jobs)
        self._completed_jobs = 0
        
        # Setup output directories
        self._setup_output_directories(options)
        
        # Start parallel processing
        self._process_jobs_parallel(jobs, options)
        
        return jobs
    
    def _process_jobs_parallel(self, jobs: List[ProcessingJob], options):
        """Process jobs with parallel execution"""
        
        # Submit jobs to the processing queue
        for job in jobs:
            self.job_queue.put((job, options))
        
        # Start worker threads
        worker_futures = []
        for worker_id in range(self.parallel_config.max_concurrent_files):
            future = self.cpu_executor.submit(self._worker_thread, worker_id)
            worker_futures.append(future)
        
        # Start progress monitoring thread
        progress_thread = threading.Thread(
            target=self._progress_monitor_thread,
            daemon=True
        )
        progress_thread.start()
        
        # Wait for all workers to complete
        concurrent.futures.wait(worker_futures)
        
        # Signal progress monitor to stop
        self.progress_queue.put(("shutdown", None))
    
    def _worker_thread(self, worker_id: int):
        """Worker thread for processing jobs"""
        self.logger.info(f"Worker {worker_id} started")
        
        while not self.is_cancelled:
            try:
                # Get next job from queue (with timeout to allow cancellation)
                try:
                    job, options = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Acquire processing slot
                if not self.resource_manager.acquire_processing_slot():
                    # Put job back and wait
                    self.job_queue.put((job, options))
                    time.sleep(0.5)
                    continue
                
                try:
                    # Process the job
                    job.worker_id = worker_id
                    job.start_time = time.time()
                    self._process_single_job_parallel(job, options)
                    
                except Exception as e:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = str(e)
                    self.logger.error(f"Job {job.job_id} failed: {e}")
                
                finally:
                    job.end_time = time.time()
                    self.resource_manager.release_processing_slot()
                    self.job_queue.task_done()
                    
                    # Update completion count
                    with self._progress_lock:
                        self._completed_jobs += 1
                        overall_progress = (self._completed_jobs / self._total_jobs) * 100
                        self.progress_queue.put(("job_completed", {
                            "job": job,
                            "overall_progress": overall_progress
                        }))
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.info(f"Worker {worker_id} finished")
    
    def _process_single_job_parallel(self, job: ProcessingJob, options):
        """Process a single job with pipeline parallelization"""
        job.status = ProcessingStatus.PROCESSING
        file_path = Path(job.file_path)
        
        self.progress_queue.put(("job_started", {"job": job, "message": f"Processing {file_path.name}"}))
        
        # Pipeline stages with potential parallelization
        if self.parallel_config.enable_pipeline_parallel:
            self._process_job_parallel_pipeline(job, options)
        else:
            self._process_job_sequential_pipeline(job, options)
        
        job.status = ProcessingStatus.COMPLETED
        job.progress = 100.0
    
    def _process_job_parallel_pipeline(self, job: ProcessingJob, options):
        """Process job with parallelized pipeline stages"""
        file_path = Path(job.file_path)
        
        # Stage 1: Audio extraction/preparation (I/O bound)
        audio_future = self.io_executor.submit(self._prepare_audio, job, options)
        
        # Wait for audio preparation
        audio_file = audio_future.result()
        job.progress = 25.0
        
        # Stage 2: Transcription (CPU bound) - submit to CPU pool
        transcription_future = self.cpu_executor.submit(
            self._transcribe_audio_parallel, job, audio_file, options
        )
        
        # Stage 3: Prepare for translation while transcribing
        if job.target_languages:
            # Wait for transcription to complete
            srt_file = transcription_future.result()
            job.progress = 75.0
            
            # Stage 4: Translation (I/O bound) - can be parallelized per language
            if len(job.target_languages) > 1:
                translation_futures = []
                for lang in job.target_languages:
                    future = self.io_executor.submit(
                        self._translate_single_language, job, srt_file, lang, options
                    )
                    translation_futures.append(future)
                
                # Wait for all translations
                concurrent.futures.wait(translation_futures)
            else:
                # Single language translation
                self._translate_subtitles_parallel(job, srt_file, options)
        else:
            # No translation needed, just wait for transcription
            transcription_future.result()
            job.progress = 100.0
    
    def _process_job_sequential_pipeline(self, job: ProcessingJob, options):
        """Process job with sequential pipeline (fallback)"""
        file_path = Path(job.file_path)
        
        # Sequential processing
        audio_file = self._prepare_audio(job, options)
        job.progress = 25.0
        
        srt_file = self._transcribe_audio_parallel(job, audio_file, options)
        job.progress = 75.0
        
        if job.target_languages:
            self._translate_subtitles_parallel(job, srt_file, options)
        
        job.progress = 100.0
    
    def _prepare_audio(self, job: ProcessingJob, options):
        """Prepare audio file (extraction and/or boosting)"""
        file_path = Path(job.file_path)
        
        # Audio extraction if needed
        if file_path.suffix.lower() == '.mp4':
            audio_file = self._extract_audio(job, options)
        else:
            audio_file = str(file_path)
        
        # Audio boosting if requested
        if options.boost_audio and job.boost_level > 1:
            audio_file = self._boost_audio(job, audio_file, options)
        
        return audio_file
    
    def _transcribe_audio_parallel(self, job: ProcessingJob, audio_file: str, options):
        """Transcribe audio using shared Whisper model"""
        if self.parallel_config.share_whisper_models:
            model, model_lock = self.resource_manager.get_whisper_model(options.whisper_model)
            
            with model_lock:
                # Transcription with shared model
                return self._transcribe_with_shared_model(job, audio_file, options, model)
        else:
            # Individual model instance (uses more memory)
            return self._transcribe_audio(job, audio_file, options)
    
    def _translate_subtitles_parallel(self, job: ProcessingJob, srt_file: str, options):
        """Translate subtitles with parallel language processing"""
        if len(job.target_languages) <= 1:
            # Single or no translation
            for lang_code in job.target_languages:
                self._translate_single_language(job, srt_file, lang_code, options)
        else:
            # Multiple languages - process in parallel
            translation_futures = []
            for lang_code in job.target_languages:
                future = self.io_executor.submit(
                    self._translate_single_language, job, srt_file, lang_code, options
                )
                translation_futures.append(future)
            
            # Wait for all translations to complete
            concurrent.futures.wait(translation_futures)
    
    def _translate_single_language(self, job: ProcessingJob, srt_file: str, lang_code: str, options):
        """Translate subtitles to a single language"""
        srt_path = Path(srt_file)
        
        try:
            output_name = f"{lang_code}_{srt_path.name}"
            output_path = Path(options.subtitle_dir or "output_subtitles") / output_name
            
            # Use translation service
            if self.translation_service:
                self.translation_service.translate_srt_file(
                    str(srt_path),
                    str(output_path),
                    lang_code
                )
            
            job.output_files.append(str(output_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to translate {job.job_id} to {lang_code}: {e}")
    
    def _progress_monitor_thread(self):
        """Monitor progress updates from worker threads"""
        while True:
            try:
                msg_type, data = self.progress_queue.get(timeout=1.0)
                
                if msg_type == "shutdown":
                    break
                elif msg_type == "job_started" and self.progress_callback:
                    job = data["job"]
                    message = data["message"]
                    self.progress_callback(None, message)
                elif msg_type == "job_completed" and self.progress_callback:
                    job = data["job"]
                    overall_progress = data["overall_progress"]
                    message = f"Completed {Path(job.file_path).name} ({job.status.value})"
                    self.progress_callback(overall_progress, message)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Progress monitor error: {e}")
    
    # Placeholder methods for actual processing (would use injected services)
    def _extract_audio(self, job: ProcessingJob, options):
        """Extract audio from video file"""
        # Implementation using audio_service
        return str(Path(job.file_path).with_suffix('.mp3'))
    
    def _boost_audio(self, job: ProcessingJob, audio_file: str, options):
        """Boost audio volume"""
        # Implementation using audio_service
        return audio_file
    
    def _transcribe_audio(self, job: ProcessingJob, audio_file: str, options):
        """Transcribe audio to SRT"""
        # Implementation using transcription_service
        return str(Path(audio_file).with_suffix('.srt'))
    
    def _transcribe_with_shared_model(self, job: ProcessingJob, audio_file: str, options, model):
        """Transcribe using shared Whisper model"""
        # Implementation using shared model
        return str(Path(audio_file).with_suffix('.srt'))
    
    def _setup_output_directories(self, options):
        """Setup output directories"""
        directories = [
            options.subtitle_dir or "output_subtitles",
            options.boosted_audio_dir or "boosted_audio"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def cancel_processing(self):
        """Cancel all processing operations"""
        self.is_cancelled = True
        
        # Update all pending/processing jobs
        for job in self.active_jobs.values():
            if job.status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
                job.status = ProcessingStatus.CANCELLED
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        with self._progress_lock:
            active_jobs = len([j for j in self.active_jobs.values() 
                             if j.status == ProcessingStatus.PROCESSING])
            
            return {
                'total_jobs': self._total_jobs,
                'completed_jobs': self._completed_jobs,
                'active_jobs': active_jobs,
                'progress_percentage': (self._completed_jobs / max(1, self._total_jobs)) * 100,
                'memory_usage_mb': self.resource_manager._memory_usage,
                'active_workers': self.resource_manager._active_jobs
            }
    
    def shutdown(self):
        """Shutdown the processing facade and clean up resources"""
        self.cancel_processing()
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)