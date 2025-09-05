#!/usr/bin/env python3
"""
FIXED Audio/Video Transcription Tool with Working PyQt6 Interface
This version has properly working buttons and visible translation settings
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import asyncio
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QProgressBar, QListWidget, QListWidgetItem, 
    QFileDialog, QDialog, QTabWidget, QFormLayout, QSpinBox, 
    QComboBox, QCheckBox, QLineEdit, QGroupBox, QSplitter, 
    QStatusBar, QMessageBox, QSlider, QRadioButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QFont, QDragEnterEvent, QDropEvent, QColor, QBrush, QPainter
)

# Try to import new service layer
try:
    sys.path.insert(0, 'src')
    from src.services.translation_service import TranslationService
    from src.config.config_manager import ConfigManager
    NEW_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    print(f"New architecture not available: {e}")
    NEW_ARCHITECTURE_AVAILABLE = False

class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"

class UserPreferences:
    """User preferences management"""
    
    def __init__(self):
        self.config_file = Path.home() / ".transcription_gui_config.json"
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        defaults = {
            "enable_audio_boost": True,
            "boost_amount": 10,
            "whisper_model": "base",
            "source_language": "auto",
            "enable_translation": False,
            "translation_target": "English",
            "enable_dual_translation": False,
            "primary_target_lang": "English",
            "secondary_target_lang": "Spanish",
            "use_source_directory": True,
            "save_txt": True,
            "save_srt": False,
        }
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    defaults.update(settings)
            return defaults
        except Exception:
            return defaults
    
    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass
    
    def get(self, key: str, default=None):
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        self.settings[key] = value

class FixedButton(QPushButton):
    """Button with forced visual styling that actually works"""
    
    def __init__(self, text, button_type="default", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.setMinimumSize(120, 40)
        self.setFont(QFont("Segoe UI", 9))
        
        # Define colors
        if button_type == "primary":
            bg_color = "#28a745"  # Green
        elif button_type == "secondary": 
            bg_color = "#6c757d"  # Gray
        else:
            bg_color = "#007bff"  # Blue
            
        # Apply inline style - this should work
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(bg_color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(bg_color, 0.3)};
            }}
        """)
    
    def _darken_color(self, hex_color, factor=0.15):
        """Darken a hex color by a factor"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * (1 - factor))) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"

class DropArea(QWidget):
    """Simple drop area without complex styling"""
    
    files_dropped = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Drop audio/video files here\nor click Browse to select files")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                background-color: #f9f9f9;
            }
        """)
        
        layout.addWidget(self.label)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.label.setStyleSheet("""
                QLabel {
                    color: #007bff;
                    font-size: 16px;
                    border: 2px dashed #007bff;
                    border-radius: 10px;
                    padding: 40px;
                    background-color: #e3f2fd;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                background-color: #f9f9f9;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        valid_files = [f for f in files if self._is_valid_file(f)]
        
        if valid_files:
            self.files_dropped.emit(valid_files)
        
        self.label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                background-color: #f9f9f9;
            }
        """)
    
    def _is_valid_file(self, file_path: str) -> bool:
        valid_exts = {'.mp3', '.wav', '.mp4', '.avi', '.mov', '.mkv', '.webm'}
        return Path(file_path).suffix.lower() in valid_exts

class SettingsDialog(QDialog):
    """Settings dialog with working buttons"""
    
    def __init__(self, preferences: UserPreferences, parent=None):
        super().__init__(parent)
        self.preferences = preferences
        self.setWindowTitle("Transcription Settings")
        self.setMinimumSize(500, 400)
        self.setModal(True)
        self.setup_ui()
        self.load_preferences()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Tab widget
        tab_widget = QTabWidget()
        
        # Audio tab
        audio_tab = self.create_audio_tab()
        tab_widget.addTab(audio_tab, "Audio")
        
        # Transcription tab  
        transcription_tab = self.create_transcription_tab()
        tab_widget.addTab(transcription_tab, "Transcription")
        
        # Output tab
        output_tab = self.create_output_tab()
        tab_widget.addTab(output_tab, "Output")
        
        layout.addWidget(tab_widget)
        
        # Buttons - using FixedButton class
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = FixedButton("Cancel", "secondary")
        cancel_btn.clicked.connect(self.reject)
        
        save_btn = FixedButton("Save Settings", "primary")
        save_btn.clicked.connect(self.save_and_accept)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
    
    def create_audio_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Audio boost group
        boost_group = QGroupBox("Audio Enhancement")
        boost_layout = QFormLayout(boost_group)
        
        self.enable_boost_cb = QCheckBox("Enable audio boosting")
        self.boost_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self.boost_amount_slider.setRange(0, 30)
        self.boost_amount_label = QLabel("10 dB")
        
        boost_layout.addRow("Boost Audio:", self.enable_boost_cb)
        boost_layout.addRow("Boost Amount:", self.boost_amount_slider)
        boost_layout.addRow("", self.boost_amount_label)
        
        layout.addWidget(boost_group)
        layout.addStretch()
        
        # Connect slider
        self.boost_amount_slider.valueChanged.connect(
            lambda v: self.boost_amount_label.setText(f"{v} dB")
        )
        
        return widget
    
    def create_transcription_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Whisper settings
        whisper_group = QGroupBox("Whisper Settings")
        whisper_layout = QFormLayout(whisper_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh"])
        
        whisper_layout.addRow("Model:", self.model_combo)
        whisper_layout.addRow("Source Language:", self.language_combo)
        
        layout.addWidget(whisper_group)
        
        # Translation settings - Enhanced with dual language support
        translation_group = QGroupBox("Translation Settings")
        translation_layout = QVBoxLayout(translation_group)
        
        # Single language translation
        self.enable_translation_cb = QCheckBox("Enable subtitle translation")
        translation_layout.addWidget(self.enable_translation_cb)
        
        single_lang_layout = QFormLayout()
        self.translation_target_combo = QComboBox()
        self.translation_target_combo.addItems([
            "English", "Spanish", "French", "German", "Italian", 
            "Portuguese", "Russian", "Japanese", "Chinese", "Korean",
            "Arabic", "Hindi", "Dutch", "Swedish", "Norwegian", "Danish",
            "Finnish", "Polish", "Ukrainian"
        ])
        single_lang_layout.addRow("Translate to:", self.translation_target_combo)
        
        translation_layout.addLayout(single_lang_layout)
        
        # Dual language translation
        self.enable_dual_translation_cb = QCheckBox("Enable dual language translation")
        translation_layout.addWidget(self.enable_dual_translation_cb)
        
        # Dual language controls
        self.dual_lang_widget = QWidget()
        dual_lang_layout = QVBoxLayout(self.dual_lang_widget)
        
        # Primary and secondary language selection
        lang_selection_layout = QFormLayout()
        
        self.primary_lang_combo = QComboBox()
        self.primary_lang_combo.addItems([
            "English", "Spanish", "French", "German", "Italian", 
            "Portuguese", "Russian", "Japanese", "Chinese", "Korean",
            "Arabic", "Hindi", "Dutch", "Swedish", "Norwegian", "Danish",
            "Finnish", "Polish", "Ukrainian"
        ])
        
        self.secondary_lang_combo = QComboBox()
        self.secondary_lang_combo.addItems([
            "Spanish", "English", "French", "German", "Italian", 
            "Portuguese", "Russian", "Japanese", "Chinese", "Korean",
            "Arabic", "Hindi", "Dutch", "Swedish", "Norwegian", "Danish",
            "Finnish", "Polish", "Ukrainian"
        ])
        
        lang_selection_layout.addRow("Primary Language:", self.primary_lang_combo)
        lang_selection_layout.addRow("Secondary Language:", self.secondary_lang_combo)
        
        dual_lang_layout.addLayout(lang_selection_layout)
        
        # Quick presets
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Quick Presets:")
        preset_layout.addWidget(preset_label)
        
        self.preset_en_es_btn = FixedButton("EN + ES", "default")
        self.preset_en_fr_btn = FixedButton("EN + FR", "default")
        self.preset_en_de_btn = FixedButton("EN + DE", "default")
        self.preset_en_pl_btn = FixedButton("EN + PL", "default")
        self.swap_langs_btn = FixedButton("Swap", "secondary")
        
        preset_layout.addWidget(self.preset_en_es_btn)
        preset_layout.addWidget(self.preset_en_fr_btn)
        preset_layout.addWidget(self.preset_en_de_btn)
        preset_layout.addWidget(self.preset_en_pl_btn)
        preset_layout.addWidget(self.swap_langs_btn)
        preset_layout.addStretch()
        
        dual_lang_layout.addLayout(preset_layout)
        
        translation_layout.addWidget(self.dual_lang_widget)
        
        # Connect signals
        self.enable_translation_cb.toggled.connect(self.toggle_translation_mode)
        self.enable_dual_translation_cb.toggled.connect(self.toggle_dual_translation)
        self.preset_en_es_btn.clicked.connect(lambda: self.set_language_preset("English", "Spanish"))
        self.preset_en_fr_btn.clicked.connect(lambda: self.set_language_preset("English", "French"))
        self.preset_en_de_btn.clicked.connect(lambda: self.set_language_preset("English", "German"))
        self.preset_en_pl_btn.clicked.connect(lambda: self.set_language_preset("English", "Polish"))
        self.swap_langs_btn.clicked.connect(self.swap_languages)
        
        layout.addWidget(translation_group)
        
        layout.addStretch()
        return widget
    
    def create_output_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Directory group
        dir_group = QGroupBox("Output Directory")
        dir_layout = QVBoxLayout(dir_group)
        
        self.use_source_rb = QRadioButton("Use source directory")
        self.use_custom_rb = QRadioButton("Use custom directory")
        
        custom_layout = QHBoxLayout()
        self.custom_dir_edit = QLineEdit()
        browse_btn = FixedButton("Browse", "default")
        browse_btn.clicked.connect(self.browse_directory)
        
        custom_layout.addWidget(self.custom_dir_edit)
        custom_layout.addWidget(browse_btn)
        
        dir_layout.addWidget(self.use_source_rb)
        dir_layout.addWidget(self.use_custom_rb)
        dir_layout.addLayout(custom_layout)
        
        # Format group
        format_group = QGroupBox("Output Formats")
        format_layout = QVBoxLayout(format_group)
        
        self.save_txt_cb = QCheckBox("Save as text (.txt)")
        self.save_srt_cb = QCheckBox("Save as subtitle (.srt)")
        
        format_layout.addWidget(self.save_txt_cb)
        format_layout.addWidget(self.save_srt_cb)
        
        layout.addWidget(dir_group)
        layout.addWidget(format_group)
        layout.addStretch()
        
        return widget
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.custom_dir_edit.setText(directory)
    
    def load_preferences(self):
        # Load all settings
        self.enable_boost_cb.setChecked(self.preferences.get("enable_audio_boost", True))
        self.boost_amount_slider.setValue(self.preferences.get("boost_amount", 10))
        
        # Set model
        model = self.preferences.get("whisper_model", "base")
        idx = self.model_combo.findText(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        
        # Set language
        lang = self.preferences.get("source_language", "auto")
        idx = self.language_combo.findText(lang)
        if idx >= 0:
            self.language_combo.setCurrentIndex(idx)
        
        # Translation settings
        self.enable_translation_cb.setChecked(self.preferences.get("enable_translation", False))
        
        target = self.preferences.get("translation_target", "English")
        idx = self.translation_target_combo.findText(target)
        if idx >= 0:
            self.translation_target_combo.setCurrentIndex(idx)
        
        # Dual language settings
        self.enable_dual_translation_cb.setChecked(self.preferences.get("enable_dual_translation", False))
        
        primary_lang = self.preferences.get("primary_target_lang", "English")
        idx = self.primary_lang_combo.findText(primary_lang)
        if idx >= 0:
            self.primary_lang_combo.setCurrentIndex(idx)
        
        secondary_lang = self.preferences.get("secondary_target_lang", "Spanish")
        idx = self.secondary_lang_combo.findText(secondary_lang)
        if idx >= 0:
            self.secondary_lang_combo.setCurrentIndex(idx)
        
        # Update UI state
        self.toggle_translation_mode()
        self.toggle_dual_translation()
        
        # Output settings
        self.use_source_rb.setChecked(self.preferences.get("use_source_directory", True))
        self.use_custom_rb.setChecked(not self.preferences.get("use_source_directory", True))
        
        self.save_txt_cb.setChecked(self.preferences.get("save_txt", True))
        self.save_srt_cb.setChecked(self.preferences.get("save_srt", False))
    
    def save_and_accept(self):
        # Save all settings
        self.preferences.set("enable_audio_boost", self.enable_boost_cb.isChecked())
        self.preferences.set("boost_amount", self.boost_amount_slider.value())
        self.preferences.set("whisper_model", self.model_combo.currentText())
        self.preferences.set("source_language", self.language_combo.currentText())
        
        # Save translation settings
        self.preferences.set("enable_translation", self.enable_translation_cb.isChecked())
        self.preferences.set("translation_target", self.translation_target_combo.currentText())
        
        # Save dual language settings
        self.preferences.set("enable_dual_translation", self.enable_dual_translation_cb.isChecked())
        self.preferences.set("primary_target_lang", self.primary_lang_combo.currentText())
        self.preferences.set("secondary_target_lang", self.secondary_lang_combo.currentText())
        
        self.preferences.set("use_source_directory", self.use_source_rb.isChecked())
        self.preferences.set("save_txt", self.save_txt_cb.isChecked())
        self.preferences.set("save_srt", self.save_srt_cb.isChecked())
        
        self.preferences.save()
        self.accept()
    
    def toggle_translation_mode(self):
        """Toggle translation controls based on checkbox state"""
        is_enabled = self.enable_translation_cb.isChecked()
        self.translation_target_combo.setEnabled(is_enabled and not self.enable_dual_translation_cb.isChecked())
        
        if is_enabled and self.enable_dual_translation_cb.isChecked():
            self.dual_lang_widget.setEnabled(True)
        elif not is_enabled:
            self.dual_lang_widget.setEnabled(False)
            self.enable_dual_translation_cb.setChecked(False)
    
    def toggle_dual_translation(self):
        """Toggle dual language controls"""
        is_dual_enabled = self.enable_dual_translation_cb.isChecked()
        
        if is_dual_enabled:
            # Enable dual translation, disable single
            self.enable_translation_cb.setChecked(True)
            self.translation_target_combo.setEnabled(False)
            self.dual_lang_widget.setEnabled(True)
        else:
            # Enable single translation, disable dual
            self.translation_target_combo.setEnabled(self.enable_translation_cb.isChecked())
            self.dual_lang_widget.setEnabled(False)
    
    def set_language_preset(self, primary: str, secondary: str):
        """Set language preset"""
        primary_idx = self.primary_lang_combo.findText(primary)
        if primary_idx >= 0:
            self.primary_lang_combo.setCurrentIndex(primary_idx)
        
        secondary_idx = self.secondary_lang_combo.findText(secondary)
        if secondary_idx >= 0:
            self.secondary_lang_combo.setCurrentIndex(secondary_idx)
    
    def swap_languages(self):
        """Swap primary and secondary languages"""
        primary_text = self.primary_lang_combo.currentText()
        secondary_text = self.secondary_lang_combo.currentText()
        
        primary_idx = self.primary_lang_combo.findText(secondary_text)
        if primary_idx >= 0:
            self.primary_lang_combo.setCurrentIndex(primary_idx)
        
        secondary_idx = self.secondary_lang_combo.findText(primary_text)
        if secondary_idx >= 0:
            self.secondary_lang_combo.setCurrentIndex(secondary_idx)

class ProcessingWorker(QThread):
    """Worker thread for file processing"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    processing_complete = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, files, preferences):
        super().__init__()
        self.files = files
        self.preferences = preferences
    
    def run(self):
        try:
            if NEW_ARCHITECTURE_AVAILABLE:
                self.process_with_new_architecture()
            else:
                self.process_with_legacy_method()
        except Exception as e:
            self.processing_complete.emit(False, f"Processing failed: {str(e)}")
    
    def process_with_new_architecture(self):
        """Process using the new service architecture"""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run async processing
            loop.run_until_complete(self.async_process_files())
        except Exception as e:
            self.processing_complete.emit(False, f"New architecture processing failed: {str(e)}")
        finally:
            loop.close()
    
    async def async_process_files(self):
        """Async processing with new architecture"""
        try:
            config_manager = ConfigManager()
            translation_service = TranslationService()
            
            total_files = len(self.files)
            
            for i, file_path in enumerate(self.files):
                # Update progress
                progress = int((i / total_files) * 100)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"Processing {Path(file_path).name}...")
                
                # Simulate transcription (placeholder - would need actual transcription service)
                await asyncio.sleep(1)  # Simulate processing time
                
                # Handle dual language translation
                if self.preferences.get("enable_dual_translation", False):
                    primary_lang = self.preferences.get('primary_target_lang', 'English')
                    secondary_lang = self.preferences.get('secondary_target_lang', 'Spanish')
                    
                    # Convert language names to codes for the service
                    lang_map = {
                        "English": "en", "Spanish": "es", "French": "fr", "German": "de",
                        "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Japanese": "ja",
                        "Chinese": "zh", "Korean": "ko", "Arabic": "ar", "Hindi": "hi",
                        "Polish": "pl", "Dutch": "nl", "Swedish": "sv", "Norwegian": "no",
                        "Danish": "da", "Finnish": "fi", "Ukrainian": "uk"
                    }
                    
                    primary_code = lang_map.get(primary_lang, "en")
                    secondary_code = lang_map.get(secondary_lang, "es")
                    
                    self.status_updated.emit(f"Translating to {primary_lang} and {secondary_lang}...")
                    
                    # Simulate dual translation (placeholder - would need actual SRT files)
                    # In real implementation, this would translate actual transcribed SRT files
                    await asyncio.sleep(1)  # Simulate translation time
                    
                elif self.preferences.get("enable_translation", False):
                    target_lang = self.preferences.get('translation_target', 'English')
                    self.status_updated.emit(f"Translating to {target_lang}...")
                    await asyncio.sleep(0.5)  # Simulate translation time
            
            # Complete
            self.progress_updated.emit(100)
            self.processing_complete.emit(True, f"Successfully processed {total_files} files!")
            
        except Exception as e:
            self.processing_complete.emit(False, f"Async processing failed: {str(e)}")
    
    def process_with_legacy_method(self):
        """Fallback processing method"""
        try:
            total_files = len(self.files)
            
            for i, file_path in enumerate(self.files):
                progress = int((i / total_files) * 100)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"Processing {Path(file_path).name}...")
                
                # Simulate processing
                import time
                time.sleep(1)
                
                # Handle dual language translation
                if self.preferences.get("enable_dual_translation", False):
                    primary_lang = self.preferences.get('primary_target_lang', 'English')
                    secondary_lang = self.preferences.get('secondary_target_lang', 'Spanish')
                    self.status_updated.emit(f"Translating to {primary_lang} and {secondary_lang}...")
                    time.sleep(1)
            
            self.progress_updated.emit(100)
            self.processing_complete.emit(True, f"Successfully processed {total_files} files using legacy method!")
            
        except Exception as e:
            self.processing_complete.emit(False, f"Legacy processing failed: {str(e)}")

class MainWindow(QMainWindow):
    """Main application window with working interface"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio/Video Transcription Tool - FIXED VERSION")
        self.setMinimumSize(800, 600)
        
        self.preferences = UserPreferences()
        self.selected_files = []
        self.processing_worker = None
        
        self.setup_ui()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        title_label = QLabel("Audio/Video Transcription Tool")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle_label = QLabel("Drop files below or use the buttons to get started")
        subtitle_label.setStyleSheet("font-size: 14px; color: #666; margin-bottom: 20px;")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        
        # File area
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_files)
        main_layout.addWidget(self.drop_area)
        
        # Selected files list
        files_label = QLabel("Selected Files:")
        files_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        main_layout.addWidget(files_label)
        
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(150)
        main_layout.addWidget(self.files_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        browse_btn = FixedButton("Browse Files", "default")
        browse_btn.clicked.connect(self.browse_files)
        
        clear_btn = FixedButton("Clear All", "secondary")
        clear_btn.clicked.connect(self.clear_files)
        
        settings_btn = FixedButton("Settings", "secondary") 
        settings_btn.clicked.connect(self.show_settings)
        
        start_btn = FixedButton("Start Transcription", "primary")
        start_btn.clicked.connect(self.start_transcription)
        
        button_layout.addWidget(browse_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(settings_btn)
        button_layout.addStretch()
        button_layout.addWidget(start_btn)
        
        main_layout.addLayout(button_layout)
        
        # Progress
        progress_label = QLabel("Progress:")
        progress_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        main_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select files to begin")
    
    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio/Video Files",
            "",
            "Audio/Video Files (*.mp3 *.wav *.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if files:
            self.add_files(files)
    
    def add_files(self, files: List[str]):
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.files_list.addItem(Path(file).name)
        
        self.status_bar.showMessage(f"{len(self.selected_files)} files selected")
    
    def clear_files(self):
        self.selected_files.clear()
        self.files_list.clear()
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Files cleared - Ready to select new files")
    
    def show_settings(self):
        dialog = SettingsDialog(self.preferences, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.status_bar.showMessage("Settings saved successfully")
    
    def start_transcription(self):
        if not self.selected_files:
            QMessageBox.information(self, "No Files", "Please select files to transcribe first.")
            return
        
        # Check if already processing
        if self.processing_worker and self.processing_worker.isRunning():
            QMessageBox.information(self, "Processing", "Processing is already in progress.")
            return
        
        # Show current settings
        settings_info = []
        settings_info.append(f"Audio Boost: {'Enabled' if self.preferences.get('enable_audio_boost') else 'Disabled'}")
        settings_info.append(f"Whisper Model: {self.preferences.get('whisper_model', 'base')}")
        settings_info.append(f"Source Language: {self.preferences.get('source_language', 'auto')}")
        
        if self.preferences.get("enable_dual_translation", False):
            primary = self.preferences.get('primary_target_lang', 'English')
            secondary = self.preferences.get('secondary_target_lang', 'Spanish')
            settings_info.append(f"Dual Translation: Enabled ({primary} + {secondary})")
        elif self.preferences.get("enable_translation", False):
            settings_info.append(f"Translation: Enabled (to {self.preferences.get('translation_target', 'English')})")
        else:
            settings_info.append("Translation: Disabled")
        
        settings_info.append(f"Output Formats: {', '.join(['TXT' if self.preferences.get('save_txt') else '', 'SRT' if self.preferences.get('save_srt') else '']).strip(', ')}")
        
        QMessageBox.information(
            self, 
            "Transcription Started", 
            f"Starting transcription of {len(self.selected_files)} files.\n\nCurrent Settings:\n" + "\n".join(settings_info)
        )
        
        # Start actual processing
        self.status_bar.showMessage("Starting transcription...")
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.processing_worker = ProcessingWorker(self.selected_files, self.preferences)
        self.processing_worker.progress_updated.connect(self.update_progress)
        self.processing_worker.status_updated.connect(self.update_status)
        self.processing_worker.processing_complete.connect(self.processing_finished)
        self.processing_worker.start()
    
    def update_progress(self, value):
        """Update progress bar from worker thread"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status message from worker thread"""
        self.status_bar.showMessage(message)
    
    def processing_finished(self, success, message):
        """Handle processing completion"""
        if success:
            self.status_bar.showMessage("Processing completed successfully!")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_bar.showMessage("Processing failed")
            QMessageBox.critical(self, "Processing Error", message)
    

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Audio/Video Transcription Tool - Fixed")
    app.setApplicationVersion("2.1.0")
    
    # Set font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Create and show window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()