"""
Comprehensive error handling and recovery system for the transcription application.
Provides structured exceptions, retry logic, and error recovery mechanisms.
"""

import logging
import time
import functools
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path


class ErrorCategory(Enum):
    """Categories of errors for better handling"""
    SYSTEM_ERROR = "system_error"          # OS, permissions, disk space
    NETWORK_ERROR = "network_error"        # API calls, internet connectivity  
    PROCESSING_ERROR = "processing_error"  # FFmpeg, Whisper failures
    VALIDATION_ERROR = "validation_error"  # Invalid input, file format
    RESOURCE_ERROR = "resource_error"      # Memory, CPU, file handles
    EXTERNAL_SERVICE_ERROR = "external_service_error"  # Translation APIs


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, processing can continue
    MEDIUM = "medium"     # Moderate issues, affects single file
    HIGH = "high"         # Major issues, affects batch processing
    CRITICAL = "critical" # Critical issues, stops all processing


@dataclass
class ErrorContext:
    """Context information for errors"""
    file_path: Optional[str] = None
    job_id: Optional[str] = None
    worker_id: Optional[int] = None
    operation: Optional[str] = None
    attempt_number: int = 1
    total_attempts: int = 1
    timestamp: float = 0.0
    additional_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.additional_data is None:
            self.additional_data = {}


class TranscriptionError(Exception):
    """Base exception for transcription application errors"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[ErrorContext] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        self.error_id = f"ERR_{int(time.time() * 1000)}"


class AudioProcessingError(TranscriptionError):
    """Errors related to audio processing (FFmpeg, etc.)"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.PROCESSING_ERROR)
        super().__init__(message, **kwargs)


class TranscriptionServiceError(TranscriptionError):
    """Errors related to transcription services (Whisper)"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.PROCESSING_ERROR)
        super().__init__(message, **kwargs)


class TranslationServiceError(TranscriptionError):
    """Errors related to translation services"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.EXTERNAL_SERVICE_ERROR)
        super().__init__(message, **kwargs)


class FileValidationError(TranscriptionError):
    """Errors related to file validation"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


class ResourceExhaustionError(TranscriptionError):
    """Errors related to resource exhaustion"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RESOURCE_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class NetworkError(TranscriptionError):
    """Network-related errors"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK_ERROR)
        super().__init__(message, **kwargs)


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_categories: List[ErrorCategory] = None
    
    def __post_init__(self):
        if self.retry_on_categories is None:
            self.retry_on_categories = [
                ErrorCategory.NETWORK_ERROR,
                ErrorCategory.EXTERNAL_SERVICE_ERROR,
                ErrorCategory.SYSTEM_ERROR  # Some system errors are transient
            ]


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, 
                 recovery_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_threshold = recovery_threshold
        
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time > self.timeout:
                    self._state = "HALF_OPEN"
                    self._success_count = 0
                else:
                    raise TranscriptionError(
                        f"Circuit breaker is OPEN for {func.__name__}",
                        category=ErrorCategory.SYSTEM_ERROR,
                        severity=ErrorSeverity.HIGH
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful operation"""
        with self._lock:
            if self._state == "HALF_OPEN":
                self._success_count += 1
                if self._success_count >= self.recovery_threshold:
                    self._state = "CLOSED"
                    self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
            elif self._state == "HALF_OPEN":
                self._state = "OPEN"  # Go back to OPEN on any failure in HALF_OPEN


class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[TranscriptionError] = []
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> TranscriptionError:
        """Convert any exception to TranscriptionError with proper categorization"""
        
        # If it's already a TranscriptionError, just update context
        if isinstance(error, TranscriptionError):
            if context:
                error.context = context
            self._record_error(error)
            return error
        
        # Categorize unknown errors
        transcription_error = self._categorize_error(error, context)
        self._record_error(transcription_error)
        return transcription_error
    
    def _categorize_error(self, error: Exception, context: Optional[ErrorContext]) -> TranscriptionError:
        """Categorize generic exceptions into TranscriptionError types"""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Network-related errors
        if any(keyword in error_message.lower() for keyword in 
               ['network', 'connection', 'timeout', 'unreachable', 'dns']):
            return NetworkError(f"{error_type}: {error_message}", 
                              context=context, original_exception=error)
        
        # File system errors
        if any(keyword in error_message.lower() for keyword in 
               ['permission', 'access', 'not found', 'directory', 'disk']):
            return TranscriptionError(f"{error_type}: {error_message}",
                                    category=ErrorCategory.SYSTEM_ERROR,
                                    context=context, original_exception=error)
        
        # Memory/resource errors
        if any(keyword in error_message.lower() for keyword in 
               ['memory', 'resource', 'limit', 'quota']):
            return ResourceExhaustionError(f"{error_type}: {error_message}",
                                         context=context, original_exception=error)
        
        # Processing errors (FFmpeg, Whisper, etc.)
        if any(keyword in error_message.lower() for keyword in 
               ['ffmpeg', 'whisper', 'codec', 'format']):
            return AudioProcessingError(f"{error_type}: {error_message}",
                                      context=context, original_exception=error)
        
        # Default to generic system error
        return TranscriptionError(f"{error_type}: {error_message}",
                                category=ErrorCategory.SYSTEM_ERROR,
                                context=context, original_exception=error)
    
    def _record_error(self, error: TranscriptionError):
        """Record error in history"""
        with self._lock:
            self.error_history.append(error)
            # Keep only recent errors (last 1000)
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
        
        # Log the error
        self.logger.error(
            f"Error {error.error_id}: {error.message} "
            f"[{error.category.value}/{error.severity.value}]",
            extra={
                'error_id': error.error_id,
                'category': error.category.value,
                'severity': error.severity.value,
                'context': error.context.__dict__ if error.context else None
            }
        )
    
    def should_retry(self, error: TranscriptionError) -> bool:
        """Determine if an error should be retried"""
        # Don't retry validation errors
        if error.category == ErrorCategory.VALIDATION_ERROR:
            return False
        
        # Don't retry if we've exceeded max attempts
        if error.context.attempt_number >= self.retry_config.max_attempts:
            return False
        
        # Check if error category is retryable
        return error.category in self.retry_config.retry_on_categories
    
    def calculate_retry_delay(self, attempt_number: int) -> float:
        """Calculate delay before retry attempt"""
        if not self.retry_config.exponential_backoff:
            delay = self.retry_config.base_delay
        else:
            delay = self.retry_config.base_delay * (2 ** (attempt_number - 1))
        
        # Apply maximum delay limit
        delay = min(delay, self.retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50% to 100% of calculated delay
        
        return delay
    
    def execute_with_retry(self, func: Callable, context: Optional[ErrorContext] = None,
                          service_name: Optional[str] = None, *args, **kwargs):
        """Execute function with retry logic and circuit breaker"""
        context = context or ErrorContext()
        context.total_attempts = self.retry_config.max_attempts
        
        circuit_breaker = None
        if service_name:
            circuit_breaker = self.get_circuit_breaker(service_name)
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            context.attempt_number = attempt
            
            try:
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                error = self.handle_error(e, context)
                
                # If this is the last attempt or shouldn't retry, raise the error
                if attempt >= self.retry_config.max_attempts or not self.should_retry(error):
                    raise error
                
                # Calculate and apply retry delay
                delay = self.calculate_retry_delay(attempt)
                self.logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{self.retry_config.max_attempts})")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise TranscriptionError("Maximum retry attempts exceeded", context=context)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        with self._lock:
            if not self.error_history:
                return {"total_errors": 0, "by_category": {}, "by_severity": {}}
            
            recent_errors = self.error_history[-100:]  # Last 100 errors
            
            by_category = {}
            by_severity = {}
            
            for error in recent_errors:
                # Count by category
                category = error.category.value
                by_category[category] = by_category.get(category, 0) + 1
                
                # Count by severity
                severity = error.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
            
            return {
                "total_errors": len(recent_errors),
                "by_category": by_category,
                "by_severity": by_severity,
                "circuit_breaker_states": {
                    name: breaker._state 
                    for name, breaker in self.circuit_breakers.items()
                }
            }


def with_error_handling(retry_config: Optional[RetryConfig] = None, 
                       service_name: Optional[str] = None):
    """Decorator for automatic error handling and retry"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create error handler (you might want to inject this)
            error_handler = ErrorHandler(retry_config)
            
            # Create context from function info
            context = ErrorContext(
                operation=func.__name__,
                additional_data={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            )
            
            return error_handler.execute_with_retry(
                func, context, service_name, *args, **kwargs
            )
        return wrapper
    return decorator


# Convenience decorators for common scenarios
def with_network_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator for network operations with retry"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retry_on_categories=[ErrorCategory.NETWORK_ERROR, ErrorCategory.EXTERNAL_SERVICE_ERROR]
    )
    return with_error_handling(config)


def with_processing_retry(max_attempts: int = 2, base_delay: float = 2.0):
    """Decorator for processing operations with retry"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retry_on_categories=[ErrorCategory.PROCESSING_ERROR, ErrorCategory.SYSTEM_ERROR]
    )
    return with_error_handling(config)


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def reset_error_handler():
    """Reset global error handler (useful for testing)"""
    global _global_error_handler
    _global_error_handler = None