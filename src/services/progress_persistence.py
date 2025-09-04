"""
Progress Persistence System
Provides job state persistence and resumption capabilities for interrupted processing.
"""

import json
import sqlite3
import logging
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import pickle
import threading
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job processing status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESUMED = "resumed"

class JobType(Enum):
    """Types of processing jobs"""
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    AUDIO_BOOST = "audio_boost"
    BATCH_PROCESS = "batch_process"
    STREAMING = "streaming"

@dataclass
class JobProgress:
    """Represents job progress state"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress_percent: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class FileProcessingState:
    """State for individual file processing"""
    file_path: str
    file_size: int
    job_id: str
    status: JobStatus
    progress_percent: float = 0.0
    chunks_total: int = 0
    chunks_completed: int = 0
    transcription_partial: Optional[str] = None
    processing_time: float = 0.0
    error_details: Optional[str] = None
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchJobState:
    """State for batch processing jobs"""
    batch_id: str
    job_type: JobType
    status: JobStatus
    total_files: int
    completed_files: int
    failed_files: int
    file_states: Dict[str, FileProcessingState] = field(default_factory=dict)
    template_id: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    output_directory: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class ProgressDatabase:
    """SQLite database for progress persistence"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Job progress table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS job_progress (
                        job_id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress_percent REAL DEFAULT 0.0,
                        current_step TEXT DEFAULT '',
                        total_steps INTEGER DEFAULT 0,
                        completed_steps INTEGER DEFAULT 0,
                        started_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        completed_at TIMESTAMP,
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3
                    )
                ''')
                
                # File processing state table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS file_processing_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        job_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress_percent REAL DEFAULT 0.0,
                        chunks_total INTEGER DEFAULT 0,
                        chunks_completed INTEGER DEFAULT 0,
                        transcription_partial TEXT,
                        processing_time REAL DEFAULT 0.0,
                        error_details TEXT,
                        checkpoint_data TEXT, -- JSON
                        updated_at TIMESTAMP NOT NULL,
                        FOREIGN KEY (job_id) REFERENCES job_progress (job_id)
                    )
                ''')
                
                # Batch job state table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS batch_job_state (
                        batch_id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        total_files INTEGER NOT NULL,
                        completed_files INTEGER DEFAULT 0,
                        failed_files INTEGER DEFAULT 0,
                        template_id TEXT,
                        settings TEXT, -- JSON
                        output_directory TEXT,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                ''')
                
                # Session recovery table for temporary data
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS session_recovery (
                        session_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        recovery_data BLOB, -- Pickled data
                        created_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NOT NULL
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_job_status ON job_progress(status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_job_updated ON job_progress(updated_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_file_job ON file_processing_state(job_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_job_state(status)')
                
                conn.commit()
                
            finally:
                conn.close()
                
    def save_job_progress(self, progress: JobProgress):
        """Save job progress to database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO job_progress 
                    (job_id, job_type, status, progress_percent, current_step, total_steps, 
                     completed_steps, started_at, updated_at, completed_at, error_message, 
                     retry_count, max_retries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    progress.job_id,
                    progress.job_type.value,
                    progress.status.value,
                    progress.progress_percent,
                    progress.current_step,
                    progress.total_steps,
                    progress.completed_steps,
                    progress.started_at.isoformat(),
                    progress.updated_at.isoformat(),
                    progress.completed_at.isoformat() if progress.completed_at else None,
                    progress.error_message,
                    progress.retry_count,
                    progress.max_retries
                ))
                conn.commit()
                
            finally:
                conn.close()
                
    def load_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Load job progress from database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('''
                    SELECT * FROM job_progress WHERE job_id = ?
                ''', (job_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                    
                return JobProgress(
                    job_id=row[0],
                    job_type=JobType(row[1]),
                    status=JobStatus(row[2]),
                    progress_percent=row[3],
                    current_step=row[4],
                    total_steps=row[5],
                    completed_steps=row[6],
                    started_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.fromisoformat(row[8]),
                    completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    error_message=row[10],
                    retry_count=row[11],
                    max_retries=row[12]
                )
                
            finally:
                conn.close()
                
    def get_incomplete_jobs(self) -> List[JobProgress]:
        """Get all incomplete jobs that can be resumed"""
        incomplete_statuses = [
            JobStatus.PENDING.value,
            JobStatus.RUNNING.value,
            JobStatus.PAUSED.value
        ]
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('''
                    SELECT * FROM job_progress 
                    WHERE status IN ({})
                    ORDER BY started_at DESC
                '''.format(','.join(['?'] * len(incomplete_statuses))), incomplete_statuses)
                
                jobs = []
                for row in cursor.fetchall():
                    jobs.append(JobProgress(
                        job_id=row[0],
                        job_type=JobType(row[1]),
                        status=JobStatus(row[2]),
                        progress_percent=row[3],
                        current_step=row[4],
                        total_steps=row[5],
                        completed_steps=row[6],
                        started_at=datetime.fromisoformat(row[7]),
                        updated_at=datetime.fromisoformat(row[8]),
                        completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                        error_message=row[10],
                        retry_count=row[11],
                        max_retries=row[12]
                    ))
                    
                return jobs
                
            finally:
                conn.close()
                
    def save_file_state(self, file_state: FileProcessingState):
        """Save file processing state"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO file_processing_state 
                    (file_path, file_size, job_id, status, progress_percent, chunks_total,
                     chunks_completed, transcription_partial, processing_time, error_details,
                     checkpoint_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_state.file_path,
                    file_state.file_size,
                    file_state.job_id,
                    file_state.status.value,
                    file_state.progress_percent,
                    file_state.chunks_total,
                    file_state.chunks_completed,
                    file_state.transcription_partial,
                    file_state.processing_time,
                    file_state.error_details,
                    json.dumps(file_state.checkpoint_data),
                    datetime.now().isoformat()
                ))
                conn.commit()
                
            finally:
                conn.close()
                
    def load_file_states(self, job_id: str) -> List[FileProcessingState]:
        """Load all file states for a job"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('''
                    SELECT * FROM file_processing_state WHERE job_id = ?
                ''', (job_id,))
                
                states = []
                for row in cursor.fetchall():
                    checkpoint_data = {}
                    try:
                        if row[11]:  # checkpoint_data
                            checkpoint_data = json.loads(row[11])
                    except:
                        pass
                        
                    states.append(FileProcessingState(
                        file_path=row[1],
                        file_size=row[2],
                        job_id=row[3],
                        status=JobStatus(row[4]),
                        progress_percent=row[5],
                        chunks_total=row[6],
                        chunks_completed=row[7],
                        transcription_partial=row[8],
                        processing_time=row[9],
                        error_details=row[10],
                        checkpoint_data=checkpoint_data
                    ))
                    
                return states
                
            finally:
                conn.close()
                
    def cleanup_old_jobs(self, older_than_days: int = 30):
        """Clean up old completed/failed jobs"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Get job IDs to delete
                cursor = conn.execute('''
                    SELECT job_id FROM job_progress 
                    WHERE status IN ('completed', 'failed', 'cancelled') 
                    AND updated_at < ?
                ''', (cutoff_date.isoformat(),))
                
                job_ids = [row[0] for row in cursor.fetchall()]
                
                if job_ids:
                    # Delete associated file states
                    placeholders = ','.join(['?'] * len(job_ids))
                    conn.execute(f'''
                        DELETE FROM file_processing_state 
                        WHERE job_id IN ({placeholders})
                    ''', job_ids)
                    
                    # Delete job progress
                    conn.execute(f'''
                        DELETE FROM job_progress 
                        WHERE job_id IN ({placeholders})
                    ''', job_ids)
                    
                    conn.commit()
                    logger.info(f"Cleaned up {len(job_ids)} old jobs")
                    
            finally:
                conn.close()

class ProgressPersistenceManager:
    """Main manager for progress persistence"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path.home() / ".local" / "share" / "transcription_tool"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.database = ProgressDatabase(self.data_dir / "progress.db")
        self.active_jobs: Dict[str, JobProgress] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        
        # Auto-save timer
        self.auto_save_interval = 30  # seconds
        self._auto_save_timer = None
        self._start_auto_save()
        
    def create_job(self, job_type: JobType, job_id: str = None) -> str:
        """Create new job and return ID"""
        if not job_id:
            job_id = str(uuid.uuid4())
            
        progress = JobProgress(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING
        )
        
        self.active_jobs[job_id] = progress
        self.database.save_job_progress(progress)
        
        logger.info(f"Created job: {job_id} ({job_type.value})")
        return job_id
        
    def start_job(self, job_id: str):
        """Mark job as started"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            job.updated_at = datetime.now()
            self.database.save_job_progress(job)
            self._notify_progress(job_id)
            
    def update_job_progress(self, 
                           job_id: str, 
                           progress_percent: float,
                           current_step: str = None,
                           completed_steps: int = None):
        """Update job progress"""
        if job_id not in self.active_jobs:
            return
            
        job = self.active_jobs[job_id]
        job.progress_percent = min(100.0, max(0.0, progress_percent))
        job.updated_at = datetime.now()
        
        if current_step:
            job.current_step = current_step
            
        if completed_steps is not None:
            job.completed_steps = completed_steps
            
        self.database.save_job_progress(job)
        self._notify_progress(job_id)
        
    def complete_job(self, job_id: str, success: bool = True, error_message: str = None):
        """Mark job as completed"""
        if job_id not in self.active_jobs:
            return
            
        job = self.active_jobs[job_id]
        job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
        job.completed_at = datetime.now()
        job.updated_at = datetime.now()
        job.progress_percent = 100.0 if success else job.progress_percent
        
        if error_message:
            job.error_message = error_message
            
        self.database.save_job_progress(job)
        self._notify_progress(job_id)
        
        # Remove from active jobs
        del self.active_jobs[job_id]
        
    def pause_job(self, job_id: str):
        """Pause job processing"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = JobStatus.PAUSED
            job.updated_at = datetime.now()
            self.database.save_job_progress(job)
            self._notify_progress(job_id)
            
    def resume_job(self, job_id: str):
        """Resume paused job"""
        # Load from database if not in active jobs
        if job_id not in self.active_jobs:
            job = self.database.load_job_progress(job_id)
            if job:
                self.active_jobs[job_id] = job
                
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = JobStatus.RESUMED
            job.updated_at = datetime.now()
            self.database.save_job_progress(job)
            self._notify_progress(job_id)
            return True
            
        return False
        
    def cancel_job(self, job_id: str):
        """Cancel job processing"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.now()
            self.database.save_job_progress(job)
            self._notify_progress(job_id)
            del self.active_jobs[job_id]
            
    def save_file_checkpoint(self, 
                            job_id: str,
                            file_path: str,
                            file_size: int,
                            progress_percent: float,
                            **checkpoint_data):
        """Save checkpoint for file processing"""
        file_state = FileProcessingState(
            file_path=file_path,
            file_size=file_size,
            job_id=job_id,
            status=JobStatus.RUNNING,
            progress_percent=progress_percent,
            checkpoint_data=checkpoint_data
        )
        
        self.database.save_file_state(file_state)
        
    def load_job_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data for job"""
        job = self.database.load_job_progress(job_id)
        if not job:
            return None
            
        file_states = self.database.load_file_states(job_id)
        
        return {
            'job_progress': job,
            'file_states': file_states,
            'can_resume': job.status in [JobStatus.PAUSED, JobStatus.RUNNING, JobStatus.PENDING]
        }
        
    def get_resumable_jobs(self) -> List[JobProgress]:
        """Get all jobs that can be resumed"""
        return self.database.get_incomplete_jobs()
        
    def register_progress_callback(self, job_id: str, callback: Callable[[JobProgress], None]):
        """Register callback for progress updates"""
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)
        
    def _notify_progress(self, job_id: str):
        """Notify progress callbacks"""
        if job_id in self.progress_callbacks and job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            for callback in self.progress_callbacks[job_id]:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
                    
    def _start_auto_save(self):
        """Start auto-save timer"""
        def auto_save():
            try:
                # Save all active jobs
                for job in self.active_jobs.values():
                    self.database.save_job_progress(job)
                    
                # Schedule next auto-save
                self._auto_save_timer = threading.Timer(self.auto_save_interval, auto_save)
                self._auto_save_timer.daemon = True
                self._auto_save_timer.start()
                
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
                
        self._auto_save_timer = threading.Timer(self.auto_save_interval, auto_save)
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()
        
    def cleanup_and_shutdown(self):
        """Cleanup resources and shutdown"""
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
            
        # Final save of all active jobs
        for job in self.active_jobs.values():
            self.database.save_job_progress(job)
            
        # Cleanup old jobs
        self.database.cleanup_old_jobs()
        
    @contextmanager
    def job_context(self, job_type: JobType, job_id: str = None):
        """Context manager for job lifecycle"""
        job_id = self.create_job(job_type, job_id)
        
        try:
            self.start_job(job_id)
            yield job_id
            self.complete_job(job_id, success=True)
            
        except Exception as e:
            self.complete_job(job_id, success=False, error_message=str(e))
            raise

class JobResumptionService:
    """Service for handling job resumption"""
    
    def __init__(self, persistence_manager: ProgressPersistenceManager):
        self.persistence = persistence_manager
        self.resumption_handlers: Dict[JobType, Callable] = {}
        
    def register_resumption_handler(self, job_type: JobType, handler: Callable):
        """Register handler for job type resumption"""
        self.resumption_handlers[job_type] = handler
        
    async def resume_job(self, job_id: str) -> bool:
        """Resume a specific job"""
        checkpoint_data = self.persistence.load_job_checkpoint(job_id)
        if not checkpoint_data or not checkpoint_data['can_resume']:
            return False
            
        job_progress = checkpoint_data['job_progress']
        handler = self.resumption_handlers.get(job_progress.job_type)
        
        if not handler:
            logger.error(f"No resumption handler for job type: {job_progress.job_type}")
            return False
            
        try:
            # Mark as resumed
            self.persistence.resume_job(job_id)
            
            # Call handler
            success = await handler(job_id, checkpoint_data)
            
            if success:
                logger.info(f"Successfully resumed job: {job_id}")
            else:
                logger.error(f"Failed to resume job: {job_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error resuming job {job_id}: {e}")
            self.persistence.complete_job(job_id, success=False, error_message=str(e))
            return False
            
    async def resume_all_jobs(self) -> Dict[str, bool]:
        """Resume all resumable jobs"""
        resumable_jobs = self.persistence.get_resumable_jobs()
        results = {}
        
        for job in resumable_jobs:
            try:
                success = await self.resume_job(job.job_id)
                results[job.job_id] = success
            except Exception as e:
                logger.error(f"Failed to resume job {job.job_id}: {e}")
                results[job.job_id] = False
                
        return results
        
    def get_resumable_job_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all resumable jobs"""
        jobs = self.persistence.get_resumable_jobs()
        
        summary = []
        for job in jobs:
            summary.append({
                'job_id': job.job_id,
                'job_type': job.job_type.value,
                'status': job.status.value,
                'progress_percent': job.progress_percent,
                'current_step': job.current_step,
                'started_at': job.started_at.isoformat(),
                'updated_at': job.updated_at.isoformat(),
                'can_resume': job.status in [JobStatus.PAUSED, JobStatus.PENDING]
            })
            
        return summary

# Global instances
persistence_manager = ProgressPersistenceManager()
resumption_service = JobResumptionService(persistence_manager)