# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an audio/video transcription and translation tool that processes media files to extract, boost, and translate subtitles. The project has evolved from a simple monolithic application to a robust, service-oriented architecture supporting both legacy and enhanced GUI interfaces, with async processing capabilities and comprehensive error handling.

## Core Architecture

### Dual Architecture Design
The project supports both legacy (`archive/boost_and_transcribe_gui.py`) and enhanced (`src/`) architectures:

- **Legacy GUI**: Original tkinter-based interface with fallback compatibility
- **Enhanced Architecture**: Modern service-oriented design with async processing, job queues, and comprehensive error handling
- **Unified Entry Point**: `main.py` provides seamless switching between GUI versions

### Service Layer Architecture (`src/services/`)
- **AudioService**: Handles FFmpeg-based audio extraction and volume boosting
- **TranscriptionService**: Manages OpenAI Whisper transcription with async support
- **TranslationService**: Provides subtitle translation using Google Translate with fallback mechanisms
- **BatchService**: Orchestrates complete media processing pipelines with parallel execution

### Core Processing Engine (`src/core/`)
- **ProcessingEngine**: Async job queue system with configurable worker pools
- **ParallelProcessor**: Handles concurrent media file processing
- **ErrorHandling**: Comprehensive error recovery and retry logic
- **EnhancedFacade**: Unified interface abstracting complexity from GUI layers

### Configuration System
- **JSON Configuration**: `config.json` for runtime settings
- **ConfigManager**: Dynamic configuration management with validation
- **Environment-specific**: Separate configurations for development/production

## Setup Commands

### Installation
```bash
# Windows
install.bat

# Linux/macOS
chmod +x install.sh
./install.sh

# Manual installation
pip install -r requirements.txt
```

### Running Applications
```bash
# Enhanced GUI (default)
python main.py

# Legacy GUI
python main.py --gui legacy

# Direct legacy GUI
python archive/boost_and_transcribe_gui.py
```

### Development Commands
```bash
# Run with enhanced GUI
python main.py --gui enhanced

# Check version
python main.py --version
```

## Key Dependencies
- **openai-whisper**: Speech-to-text transcription engine
- **ffmpeg-python**: Audio/video processing and manipulation
- **PyQt6**: Modern GUI framework for enhanced interface
- **deep_translator**: Google Translate integration for subtitle translation
- **pysrt**: SRT subtitle file parsing and manipulation
- **tkinter**: Legacy GUI framework (Python built-in)

## Processing Pipeline Architecture

### Data Flow
```
Input Media (MP3/MP4) → Audio Extraction → Audio Boosting → Whisper Transcription → SRT Generation → Optional Translation → Output Files
```

### Job Processing System
1. **Job Creation**: Media files queued as processing jobs
2. **Worker Pool**: Configurable concurrent processing (default: 3 workers)
3. **Status Tracking**: Real-time progress updates with callback system
4. **Error Recovery**: Automatic retry logic with configurable attempts
5. **Result Aggregation**: Comprehensive output with success/failure reporting

### File Processing Logic
- **Audio Extraction**: MP4 → MP3 conversion using FFmpeg with quality preservation
- **Volume Boosting**: Configurable amplification (1x-10x) for improved transcription accuracy
- **Transcription**: Whisper "medium" model with task-specific optimization
- **Translation**: Multi-stage translation with source language auto-detection

## Output Structure
- **Subtitles**: `output_subtitles/` - Generated SRT files
- **Boosted Audio**: `boosted_audio/` - Volume-enhanced audio files
- **Logs**: `logs/` - Application and processing logs
- **Temp Files**: `temp/` - Temporary processing artifacts

### File Naming Conventions
- Boosted audio: `boosted_[original_name].mp3`
- Subtitles: `[original_name].srt`
- Translated subtitles: `[LANG_CODE]_[original_name].srt`

## Error Handling Strategy

### Multi-Layer Fallback System
1. **New Architecture First**: Attempt processing with enhanced services
2. **Graceful Degradation**: Fall back to legacy processing on failure
3. **Service Isolation**: Individual service failures don't cascade
4. **User Feedback**: Clear error messages with actionable recovery steps

### Retry Logic
- **Configurable Attempts**: Default 2 retries per job
- **Exponential Backoff**: Progressive delay between retry attempts
- **Context Preservation**: Maintain job state across retry cycles

## Configuration Management

### Key Settings (`config.json`)
- **Processing**: `max_concurrent_jobs`, `job_timeout`, `retry_attempts`
- **Audio**: `default_boost_level`, `output_quality`, `codec`
- **Transcription**: `model`, `task`, `output_format`
- **Directories**: `output_dir`, `boosted_audio_dir`, `log_dir`

### Runtime Configuration
- Settings loaded at application startup
- Dynamic updates supported through ConfigManager
- Environment-specific overrides available

## GUI Architecture Comparison

### Legacy GUI (`archive/boost_and_transcribe_gui.py`)
- **Framework**: tkinter with threading for non-blocking operations
- **Processing**: Direct FFmpeg/Whisper calls with subprocess
- **Error Handling**: Basic exception catching with user notifications

### Enhanced GUI (`src/ui/enhanced_modern_gui.py`)
- **Framework**: Modern interface with progress tracking
- **Processing**: Service layer integration with async operations
- **Features**: Real-time progress updates, job management, comprehensive status reporting

## Development Guidelines

### Adding New Features
1. **Service First**: Implement business logic in appropriate service class
2. **Async Support**: Use async/await patterns for I/O operations  
3. **Error Handling**: Implement comprehensive error recovery
4. **Configuration**: Add new settings to `config.json` with validation
5. **Testing**: Verify both legacy and enhanced GUI compatibility

### Service Integration Pattern
```python
# Register new job handlers with the processing engine
engine = await get_processing_engine()
engine.register_job_handler(JobType.NEW_FEATURE, handler_function)
```

### Backward Compatibility
- Legacy GUI must continue functioning without new architecture
- New features should gracefully degrade when services unavailable
- Configuration changes must not break existing setups