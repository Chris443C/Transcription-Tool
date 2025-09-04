"""Services module initialization and registration."""

import logging
from typing import Optional

from ..core.processing_engine import ProcessingEngine, get_processing_engine
from .audio_service import register_audio_handlers
from .transcription_service import register_transcription_handlers
from .translation_service import register_translation_handlers
from .batch_service import register_batch_handlers

logger = logging.getLogger(__name__)


async def initialize_services() -> ProcessingEngine:
    """Initialize all services and register job handlers."""
    logger.info("Initializing services")
    
    # Get the processing engine
    engine = await get_processing_engine()
    
    # Register all job handlers
    register_audio_handlers(engine)
    register_transcription_handlers(engine)
    register_translation_handlers(engine)
    register_batch_handlers(engine)
    
    logger.info("All services initialized and handlers registered")
    return engine


# Export main service classes for direct use
from .audio_service import AudioService
from .transcription_service import TranscriptionService
from .translation_service import TranslationService
from .batch_service import BatchProcessingService

__all__ = [
    'AudioService',
    'TranscriptionService', 
    'TranslationService',
    'BatchProcessingService',
    'initialize_services',
]