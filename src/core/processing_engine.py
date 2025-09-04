"""Async processing engine with job queue for media processing tasks."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import json

from ..config.configuration import get_config

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Types of processing jobs."""
    AUDIO_EXTRACTION = "audio_extraction"
    AUDIO_BOOST = "audio_boost"
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    DUAL_TRANSLATION = "dual_translation"
    BATCH_PROCESS = "batch_process"


@dataclass
class JobResult:
    """Result of a job execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    output_files: List[str] = field(default_factory=list)


@dataclass
class Job:
    """Processing job definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType = JobType.BATCH_PROCESS
    input_data: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[JobResult] = None
    progress: float = 0.0
    progress_message: str = ""
    retry_count: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "retry_count": self.retry_count,
            "duration": self.duration,
        }


class ProcessingEngine:
    """Async processing engine with job queue management."""
    
    def __init__(self):
        self.config = get_config()
        self.jobs: Dict[str, Job] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.status_callbacks: List[Callable[[Job], None]] = []
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the processing engine and worker tasks."""
        if self.is_running:
            logger.warning("Processing engine is already running")
            return
        
        self.is_running = True
        logger.info("Starting processing engine")
        
        # Start worker tasks
        max_workers = self.config.processing.max_concurrent_jobs
        for i in range(max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {max_workers} worker tasks")
    
    async def stop(self) -> None:
        """Stop the processing engine and all workers."""
        if not self.is_running:
            return
        
        logger.info("Stopping processing engine")
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel all running jobs
        for task in self.running_jobs.values():
            task.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        self.running_jobs.clear()
        logger.info("Processing engine stopped")
    
    def add_status_callback(self, callback: Callable[[Job], None]) -> None:
        """Add a callback to be notified of job status changes."""
        self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[Job], None]) -> None:
        """Remove a status callback."""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    async def submit_job(self, job: Job) -> str:
        """Submit a job to the processing queue."""
        self.jobs[job.id] = job
        await self.job_queue.put(job)
        logger.info(f"Job {job.id} ({job.job_type.value}) submitted to queue")
        return job.id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def get_jobs(self, status_filter: Optional[JobStatus] = None) -> List[Job]:
        """Get all jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        # Cancel running job
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        await self._notify_status_change(job)
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    async def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Optional[JobResult]:
        """Wait for a job to complete and return its result."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        start_time = asyncio.get_event_loop().time()
        
        while job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for job {job_id}")
                return None
            
            await asyncio.sleep(0.1)
        
        return job.result
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get processing queue status information."""
        pending_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.PENDING])
        running_jobs = len(self.running_jobs)
        
        return {
            "is_running": self.is_running,
            "total_jobs": len(self.jobs),
            "pending_jobs": pending_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
            "failed_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED]),
            "queue_size": self.job_queue.qsize(),
            "max_workers": self.config.processing.max_concurrent_jobs,
        }
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task that processes jobs from the queue."""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Wait for job with timeout to allow shutdown
                job = await asyncio.wait_for(
                    self.job_queue.get(),
                    timeout=1.0
                )
                
                if job.status != JobStatus.PENDING:
                    continue
                
                await self._execute_job(job, worker_name)
                
            except asyncio.TimeoutError:
                # Timeout waiting for job, check if we should shutdown
                if self._shutdown_event.is_set():
                    break
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _execute_job(self, job: Job, worker_name: str) -> None:
        """Execute a single job."""
        logger.info(f"Worker {worker_name} executing job {job.id} ({job.job_type.value})")
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        await self._notify_status_change(job)
        
        # Create execution task
        task = asyncio.create_task(self._run_job_with_timeout(job))
        self.running_jobs[job.id] = task
        
        try:
            result = await task
            job.result = result
            job.status = JobStatus.COMPLETED
            logger.info(f"Job {job.id} completed successfully")
            
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            logger.info(f"Job {job.id} was cancelled")
            
        except Exception as e:
            job.result = JobResult(success=False, error=str(e))
            
            # Retry logic
            if job.retry_count < self.config.processing.retry_attempts:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                await self.job_queue.put(job)
                logger.warning(f"Job {job.id} failed, retrying ({job.retry_count}/{self.config.processing.retry_attempts})")
            else:
                job.status = JobStatus.FAILED
                logger.error(f"Job {job.id} failed after {job.retry_count} retries: {e}")
        
        finally:
            # Clean up
            job.completed_at = datetime.now()
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]
            await self._notify_status_change(job)
    
    async def _run_job_with_timeout(self, job: Job) -> JobResult:
        """Run a job with timeout."""
        timeout = self.config.processing.job_timeout
        
        try:
            return await asyncio.wait_for(
                self._execute_job_logic(job),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise Exception(f"Job timed out after {timeout} seconds")
    
    async def _execute_job_logic(self, job: Job) -> JobResult:
        """Execute the actual job logic based on job type."""
        # This is a placeholder - actual implementation will be handled by services
        # The services will register their execution handlers with the engine
        
        if hasattr(self, f'_execute_{job.job_type.value}'):
            handler = getattr(self, f'_execute_{job.job_type.value}')
            return await handler(job)
        else:
            raise NotImplementedError(f"No handler for job type: {job.job_type.value}")
    
    async def _notify_status_change(self, job: Job) -> None:
        """Notify all callbacks of job status change."""
        for callback in self.status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job)
                else:
                    callback(job)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def register_job_handler(self, job_type: JobType, handler: Callable) -> None:
        """Register a handler for a specific job type."""
        setattr(self, f'_execute_{job_type.value}', handler)
        logger.info(f"Registered handler for job type: {job_type.value}")
    
    async def update_job_progress(self, job_id: str, progress: float, message: str = "") -> None:
        """Update job progress."""
        job = self.jobs.get(job_id)
        if job:
            job.progress = max(0.0, min(1.0, progress))
            job.progress_message = message
            await self._notify_status_change(job)


# Global processing engine instance
_processing_engine: Optional[ProcessingEngine] = None


async def get_processing_engine() -> ProcessingEngine:
    """Get the global processing engine instance."""
    global _processing_engine
    if _processing_engine is None:
        _processing_engine = ProcessingEngine()
        await _processing_engine.start()
    return _processing_engine


async def create_batch_job(
    input_files: List[str],
    output_dir: str,
    boosted_audio_dir: str,
    boost_audio: bool = True,
    extract_subtitles: bool = True,
    boost_level: int = 3,
    translate_subtitles: bool = False,
    src_lang: str = "auto",
    dest_lang: str = "en"
) -> Job:
    """Create a batch processing job."""
    job_data = {
        "input_files": input_files,
        "output_dir": output_dir,
        "boosted_audio_dir": boosted_audio_dir,
        "boost_audio": boost_audio,
        "extract_subtitles": extract_subtitles,
        "boost_level": boost_level,
        "translate_subtitles": translate_subtitles,
        "src_lang": src_lang,
        "dest_lang": dest_lang,
    }
    
    return Job(
        job_type=JobType.BATCH_PROCESS,
        input_data=job_data
    )