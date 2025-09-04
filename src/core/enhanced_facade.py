"""
Enhanced processing facade integrating parallel processing and error handling.
Combines all improvements into a single, production-ready interface.
"""

import logging
import asyncio
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import queue

from ..config.config_manager import ConfigManager
from .parallel_processor import (
    ParallelProcessingFacade, ParallelProcessingConfig, ProcessingJob, ProcessingStatus
)
from .error_handling import (
    ErrorHandler, TranscriptionError, ErrorContext, RetryConfig,
    AudioProcessingError, TranscriptionServiceError, TranslationServiceError
)


@dataclass
class ProcessingOptions:
    """Enhanced processing options with error handling configuration"""
    boost_audio: bool = True
    boost_level: int = 1
    target_languages: List[str] = None
    output_dir: Optional[str] = None
    subtitle_dir: Optional[str] = None
    boosted_audio_dir: Optional[str] = None
    whisper_model: str = "medium"
    
    # Parallel processing options
    max_concurrent_files: int = 2
    enable_pipeline_parallel: bool = True
    share_whisper_models: bool = True
    
    # Error handling options
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True
    fail_fast: bool = False
    
    def __post_init__(self):
        if self.target_languages is None:
            self.target_languages = []


class EnhancedProcessingFacade:
    """
    Production-ready processing facade combining parallel processing,
    error handling, progress tracking, and resource management.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None,
                 progress_callback: Optional[Callable] = None):
        
        self.config_manager = config_manager or ConfigManager()
        self.progress_callback = progress_callback
        
        # Initialize error handling
        self.error_handler = ErrorHandler(
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                exponential_backoff=True
            )
        )
        
        # Initialize parallel processing
        self.parallel_config = ParallelProcessingConfig(
            max_concurrent_files=2,
            enable_pipeline_parallel=True,
            share_whisper_models=True
        )
        
        self.parallel_facade = ParallelProcessingFacade(
            config_manager=self.config_manager,
            progress_callback=self._enhanced_progress_callback,
            parallel_config=self.parallel_config
        )
        
        # Services will be injected
        self.audio_service = None
        self.transcription_service = None
        self.translation_service = None
        
        # Enhanced progress tracking
        self._processing_stats = {
            'total_files': 0,
            'completed_files': 0,
            'failed_files': 0,
            'errors': [],
            'start_time': None,
            'estimated_completion': None
        }
        
        self.logger = logging.getLogger(__name__)
    
    def set_services(self, audio_service, transcription_service, translation_service):
        """Inject service dependencies"""
        self.audio_service = audio_service
        self.transcription_service = transcription_service
        self.translation_service = translation_service
        
        # Pass services to parallel facade
        self.parallel_facade.set_services(
            audio_service, transcription_service, translation_service
        )
    
    def process_files(self, file_paths: List[str], options: ProcessingOptions) -> List[ProcessingJob]:
        """
        Process files with comprehensive error handling and parallel processing.
        
        Args:
            file_paths: List of file paths to process
            options: Processing configuration options
            
        Returns:
            List of ProcessingJob objects with results and error information
        """
        
        # Update configurations from options
        self._update_configurations(options)
        
        # Initialize processing statistics
        self._initialize_stats(file_paths)
        
        # Validate input files
        validated_files = self._validate_input_files(file_paths, options)
        
        if not validated_files:
            self.logger.warning("No valid files to process")
            return []
        
        # Process files with error handling
        try:
            jobs = self._process_with_error_handling(validated_files, options)
            self._finalize_processing(jobs)
            return jobs
            
        except Exception as e:
            error = self.error_handler.handle_error(e, ErrorContext(operation="process_files"))
            self.logger.critical(f"Critical processing error: {error.message}")
            
            if options.fail_fast:
                raise error
            
            # Return empty jobs list with error information
            return self._create_error_jobs(file_paths, error)
    
    def _update_configurations(self, options: ProcessingOptions):
        """Update internal configurations from processing options"""
        
        # Update parallel processing config
        self.parallel_config.max_concurrent_files = options.max_concurrent_files
        self.parallel_config.enable_pipeline_parallel = options.enable_pipeline_parallel
        self.parallel_config.share_whisper_models = options.share_whisper_models
        
        # Update error handling config
        self.error_handler.retry_config.max_attempts = options.max_retries
        self.error_handler.retry_config.base_delay = options.retry_delay
        
        # Update parallel facade
        self.parallel_facade.parallel_config = self.parallel_config
        self.parallel_facade.resource_manager = self.parallel_facade.resource_manager.__class__(
            self.parallel_config
        )
    
    def _initialize_stats(self, file_paths: List[str]):
        """Initialize processing statistics"""
        self._processing_stats = {
            'total_files': len(file_paths),
            'completed_files': 0,
            'failed_files': 0,
            'errors': [],
            'start_time': time.time(),
            'estimated_completion': None
        }
    
    def _validate_input_files(self, file_paths: List[str], options: ProcessingOptions) -> List[str]:
        """Validate input files and return list of valid files"""
        valid_files = []
        
        for file_path in file_paths:
            try:
                self._validate_single_file(file_path)
                valid_files.append(file_path)
                
            except FileValidationError as e:
                self.logger.warning(f"Invalid file {file_path}: {e.message}")
                self._processing_stats['errors'].append({
                    'file': file_path,
                    'error': e.message,
                    'type': 'validation'
                })
                
                if not options.continue_on_error:
                    raise e
        
        return valid_files
    
    def _validate_single_file(self, file_path: str):
        """Validate a single file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileValidationError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {file_path}")
        
        if path.stat().st_size == 0:
            raise FileValidationError(f"File is empty: {file_path}")
        
        # Check file extension
        supported_formats = ['.mp3', '.mp4', '.wav', '.m4a']
        if path.suffix.lower() not in supported_formats:
            raise FileValidationError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
    
    def _process_with_error_handling(self, file_paths: List[str], 
                                   options: ProcessingOptions) -> List[ProcessingJob]:
        """Process files with comprehensive error handling"""
        
        context = ErrorContext(
            operation="batch_processing",
            additional_data={'file_count': len(file_paths)}
        )
        
        try:
            # Use error handler's retry mechanism for the entire batch
            return self.error_handler.execute_with_retry(
                self._execute_parallel_processing,
                context=context,
                service_name="batch_processor",
                file_paths,
                options
            )
            
        except TranscriptionError as e:
            self.logger.error(f"Batch processing failed: {e.message}")
            
            # If continue_on_error is True, try to process individual files
            if options.continue_on_error and len(file_paths) > 1:
                return self._process_files_individually(file_paths, options)
            else:
                raise e
    
    def _execute_parallel_processing(self, file_paths: List[str], 
                                   options: ProcessingOptions) -> List[ProcessingJob]:
        """Execute parallel processing (wrapped for retry mechanism)"""
        return self.parallel_facade.process_files_parallel(file_paths, options)
    
    def _process_files_individually(self, file_paths: List[str], 
                                  options: ProcessingOptions) -> List[ProcessingJob]:
        """Process files one by one when batch processing fails"""
        self.logger.info("Falling back to individual file processing")
        
        all_jobs = []
        
        for file_path in file_paths:
            try:
                jobs = self._process_single_file_with_retry(file_path, options)
                all_jobs.extend(jobs)
                
            except Exception as e:
                # Create failed job entry
                error = self.error_handler.handle_error(e)
                failed_job = ProcessingJob(
                    job_id=f"failed_{int(time.time() * 1000)}",
                    file_path=file_path,
                    boost_level=options.boost_level,
                    target_languages=options.target_languages,
                    status=ProcessingStatus.FAILED,
                    error_message=error.message
                )
                all_jobs.append(failed_job)
                
                self._processing_stats['failed_files'] += 1
                self._processing_stats['errors'].append({
                    'file': file_path,
                    'error': error.message,
                    'type': 'processing'
                })
        
        return all_jobs
    
    def _process_single_file_with_retry(self, file_path: str, 
                                      options: ProcessingOptions) -> List[ProcessingJob]:
        """Process a single file with retry logic"""
        context = ErrorContext(
            file_path=file_path,
            operation="single_file_processing"
        )
        
        return self.error_handler.execute_with_retry(
            self._execute_parallel_processing,
            context=context,
            service_name="single_file_processor",
            [file_path],
            options
        )
    
    def _create_error_jobs(self, file_paths: List[str], error: TranscriptionError) -> List[ProcessingJob]:
        """Create error job entries when processing fails completely"""
        error_jobs = []
        
        for file_path in file_paths:
            job = ProcessingJob(
                job_id=f"error_{int(time.time() * 1000)}",
                file_path=file_path,
                boost_level=1,
                target_languages=[],
                status=ProcessingStatus.FAILED,
                error_message=error.message
            )
            error_jobs.append(job)
        
        return error_jobs
    
    def _enhanced_progress_callback(self, progress: Optional[float], message: str):
        """Enhanced progress callback with error tracking"""
        
        # Update processing statistics
        if "Completed" in message:
            self._processing_stats['completed_files'] += 1
        elif "Failed" in message:
            self._processing_stats['failed_files'] += 1
        
        # Calculate estimated completion time
        if self._processing_stats['completed_files'] > 0:
            elapsed = time.time() - self._processing_stats['start_time']
            avg_time_per_file = elapsed / self._processing_stats['completed_files']
            remaining_files = (self._processing_stats['total_files'] - 
                             self._processing_stats['completed_files'])
            
            if remaining_files > 0:
                estimated_remaining = avg_time_per_file * remaining_files
                self._processing_stats['estimated_completion'] = time.time() + estimated_remaining
        
        # Call original progress callback if provided
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def _finalize_processing(self, jobs: List[ProcessingJob]):
        """Finalize processing and generate summary"""
        
        completed_jobs = [j for j in jobs if j.status == ProcessingStatus.COMPLETED]
        failed_jobs = [j for j in jobs if j.status == ProcessingStatus.FAILED]
        
        # Update final statistics
        self._processing_stats['completed_files'] = len(completed_jobs)
        self._processing_stats['failed_files'] = len(failed_jobs)
        
        # Log processing summary
        total_time = time.time() - self._processing_stats['start_time']
        self.logger.info(
            f"Processing completed: {len(completed_jobs)} successful, "
            f"{len(failed_jobs)} failed in {total_time:.2f}s"
        )
        
        # Generate error summary
        error_summary = self.error_handler.get_error_summary()
        if error_summary['total_errors'] > 0:
            self.logger.warning(f"Error summary: {error_summary}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self._processing_stats.copy()
        
        # Add error handler statistics
        stats['error_summary'] = self.error_handler.get_error_summary()
        
        # Add parallel processing statistics
        stats['parallel_stats'] = self.parallel_facade.get_processing_stats()
        
        # Calculate success rate
        if stats['total_files'] > 0:
            stats['success_rate'] = (stats['completed_files'] / stats['total_files']) * 100
        else:
            stats['success_rate'] = 0
        
        return stats
    
    def cancel_processing(self):
        """Cancel all processing operations"""
        self.parallel_facade.cancel_processing()
        
        # Update statistics
        if self.progress_callback:
            self.progress_callback(None, "Processing cancelled by user")
    
    def get_supported_formats(self) -> List[str]:
        """Get supported file formats"""
        return ['.mp3', '.mp4', '.wav', '.m4a']
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate current configuration"""
        issues = {'errors': [], 'warnings': []}
        
        # Validate parallel processing config
        if self.parallel_config.max_concurrent_files < 1:
            issues['errors'].append("Max concurrent files must be at least 1")
        
        if self.parallel_config.max_concurrent_files > 8:
            issues['warnings'].append("High concurrency may impact system performance")
        
        # Validate services
        if not all([self.audio_service, self.transcription_service, self.translation_service]):
            issues['errors'].append("Required services not properly configured")
        
        # Validate configuration manager
        config_validation = self.config_manager.validate_config()
        issues['errors'].extend(config_validation['errors'])
        issues['warnings'].extend(config_validation['warnings'])
        
        return issues
    
    def shutdown(self):
        """Shutdown the processing facade and clean up resources"""
        self.parallel_facade.shutdown()
        self.logger.info("Enhanced processing facade shut down")