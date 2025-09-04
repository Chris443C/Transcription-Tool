# Dual Language Translation Feature

This guide explains the new dual language translation functionality that allows the transcription tool to output subtitles in two different languages simultaneously.

## Overview

The dual language translation feature enables users to:
- Generate subtitles in two target languages from a single source
- Configure primary and secondary language preferences
- Maintain backward compatibility with single language mode
- Use both enhanced and legacy GUI interfaces

## Architecture

### Enhanced Architecture (src/)
- **Translation Service**: `src/services/translation_service.py`
- **Configuration**: `src/config/configuration.py` and `src/config/config_manager.py`
- **GUI**: `src/ui/enhanced_modern_gui.py`
- **Processing Engine**: `src/core/processing_engine.py`

### Legacy Architecture (archive/)
- **GUI**: `archive/boost_and_transcribe_gui.py`
- **Backward Compatibility**: Integrated with new architecture when available

## Configuration

### Configuration File (`config.json`)
```json
{
  "translation": {
    "default_source_lang": "auto",
    "default_target_lang": "en",
    "enable_dual_translation": false,
    "primary_target_lang": "en",
    "secondary_target_lang": "es"
  }
}
```

### Configuration Options
- `enable_dual_translation`: Boolean to enable/disable dual language mode
- `primary_target_lang`: Primary target language code (e.g., "en" for English)
- `secondary_target_lang`: Secondary target language code (e.g., "es" for Spanish)

### Supported Languages
The system supports all languages available through Google Translate:
- English (en), Spanish (es), French (fr), German (de)
- Italian (it), Portuguese (pt), Russian (ru), Chinese (zh)
- Japanese (ja), Korean (ko), Arabic (ar), Hindi (hi)
- Dutch (nl), Swedish (sv), Norwegian (no), Danish (da)
- Finnish (fi), Ukrainian (uk), Polish (pl)

## Usage

### Enhanced GUI Interface

1. **Enable Dual Translation**:
   - Check "Enable Dual Language Translation" in the Translation Configuration section
   - The interface switches to dual language mode

2. **Select Languages**:
   - Choose Primary Language from dropdown
   - Choose Secondary Language from dropdown
   - Use the "Swap" button to exchange primary/secondary languages

3. **Quick Presets**:
   - "Dual EN+ES": English + Spanish
   - "Dual EN+FR": English + French
   - "Dual EN+DE": English + German

4. **Processing**:
   - Select media files as usual
   - Configure audio and transcription settings
   - Click "Start Processing"
   - The system generates subtitles in both languages automatically

### Legacy GUI Interface

1. **Enable Translation and Dual Mode**:
   - Check "Translate Subtitles"
   - Check "Dual Language Mode"

2. **Configure Languages**:
   - Set source language in "From" dropdown
   - Select primary language in "Primary" dropdown
   - Select secondary language in "Secondary" dropdown

3. **Processing**:
   - Select files and configure settings as usual
   - Start processing to generate dual language subtitles

### Programmatic Usage

```python
import asyncio
from src.services.translation_service import TranslationService
from src.config.config_manager import ConfigManager

async def dual_translate_example():
    # Configure dual translation
    config_manager = ConfigManager()
    config_manager.set_dual_translation(True, "en", "es")
    
    # Initialize service
    service = TranslationService()
    
    # Define target languages
    target_languages = [
        ("output_en.srt", "en"),
        ("output_es.srt", "es")
    ]
    
    # Translate to multiple languages
    results = await service.translate_subtitles_to_multiple_languages(
        input_file="input.srt",
        target_languages=target_languages,
        src_lang="auto"
    )
    
    return results

# Run the example
results = asyncio.run(dual_translate_example())
```

### Legacy Function Compatibility

```python
from src.services.translation_service import translate_srt_dual_language

# Legacy-style dual translation
output_files = [
    ("output_en.srt", "en"),
    ("output_es.srt", "es")
]

success_results = await translate_srt_dual_language(
    "input.srt", 
    output_files, 
    "auto"
)
```

## Output File Naming

### Enhanced Mode
When dual translation is enabled, output files are named:
- `{primary_lang}_{original_filename}.srt` (e.g., "en_video.srt")
- `{secondary_lang}_{original_filename}.srt` (e.g., "es_video.srt")

### Legacy Mode Compatibility
The system maintains compatibility with existing naming conventions and can generate files using the configured naming pattern.

## Error Handling

### Service Availability
- The system checks for translation service availability
- Falls back to legacy translation methods if enhanced service fails
- Provides clear error messages for missing dependencies

### Language Validation
- Validates that primary and secondary languages are different in dual mode
- Checks that selected languages are supported
- Provides helpful error messages for configuration issues

### Processing Errors
- Individual language translation failures don't stop the entire process
- Failed translations are logged with specific error details
- Partial success is reported (e.g., "1 of 2 languages translated successfully")

## Performance Considerations

### Parallel Processing
- Both language translations run concurrently when possible
- Progress reporting includes overall completion across all languages
- Memory usage is optimized by processing files sequentially

### Resource Management
- Translation requests are throttled to respect API limits
- Temporary files are cleaned up automatically
- Long-running processes can be cancelled gracefully

## Troubleshooting

### Common Issues

1. **Translation Service Not Available**
   ```
   Error: Translation service not available
   ```
   **Solution**: Install required dependencies:
   ```bash
   pip install deep_translator pysrt
   ```

2. **Same Language Selected for Both**
   ```
   Error: Primary and secondary languages must be different
   ```
   **Solution**: Choose different languages for primary and secondary options

3. **Network Connection Issues**
   ```
   Error: Translation failed - network error
   ```
   **Solution**: Check internet connection; translation requires online access

### Debug Mode

Enable debug logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing the Feature

Run the demo script to test functionality:
```bash
python dual_translation_demo.py
```

## Migration Guide

### From Single to Dual Language

1. **Backup existing configurations**
2. **Update config.json** with new dual language settings
3. **Test with sample files** before processing important content
4. **Update any custom scripts** to use new API methods

### Backward Compatibility

The system maintains full backward compatibility:
- Existing single language workflows continue to work
- Legacy GUI functions are preserved
- Configuration files are automatically upgraded

## API Reference

### TranslationService Methods

#### `translate_subtitles_to_multiple_languages(input_file, target_languages, src_lang, progress_callback)`
Translate a subtitle file to multiple target languages.

**Parameters:**
- `input_file` (str): Path to input SRT file
- `target_languages` (List[tuple]): List of (output_file, dest_lang) tuples
- `src_lang` (str): Source language code (default: "auto")
- `progress_callback` (callable): Optional progress callback function

**Returns:**
- `List[Dict]`: Results for each language translation

#### `is_dual_translation_enabled()`
Check if dual translation mode is enabled.

**Returns:**
- `bool`: True if dual translation is enabled

### ConfigManager Methods

#### `set_dual_translation(enabled, primary_lang, secondary_lang)`
Configure dual translation settings.

**Parameters:**
- `enabled` (bool): Enable/disable dual translation
- `primary_lang` (str): Primary target language code
- `secondary_lang` (str): Secondary target language code

#### `get_translation_languages()`
Get current primary and secondary language settings.

**Returns:**
- `tuple`: (primary_lang, secondary_lang)

## Future Enhancements

### Planned Features
- Support for more than 2 simultaneous languages
- Custom output file naming patterns
- Batch processing optimizations
- Language detection improvements

### Integration Possibilities
- REST API for web-based interfaces
- Command-line interface enhancements
- Docker container support
- Cloud translation service integration

## Contributing

To contribute to the dual language feature:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Update documentation**
5. **Submit a pull request**

### Testing Guidelines
- Test with various language combinations
- Verify backward compatibility
- Check error handling paths
- Validate configuration changes

## License

This feature is part of the transcription tool and follows the same license terms as the main project.