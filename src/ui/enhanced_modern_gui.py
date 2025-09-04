"""
Enhanced modern GUI with parallel processing, error handling, and improved UX.
Production-ready interface for the transcription application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

from ..core.enhanced_facade import EnhancedProcessingFacade, ProcessingOptions, ProcessingJob
from ..core.parallel_processor import ProcessingStatus
from ..config.config_manager import ConfigManager


class EnhancedTranscriptionGUI:
    """
    Enhanced GUI with parallel processing, comprehensive error handling,
    and production-ready user experience.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio/Video Transcription Tool - Enhanced v2.1")
        self.root.geometry("900x800")
        self.root.minsize(800, 600)
        
        # Initialize configuration and facade
        self.config_manager = ConfigManager()
        self.processing_facade = EnhancedProcessingFacade(
            config_manager=self.config_manager,
            progress_callback=self._update_progress
        )
        
        # Load UI preferences
        self._load_ui_preferences()
        
        # UI state
        self.selected_files: List[str] = []
        self.current_jobs: List[ProcessingJob] = []
        self.is_processing = False
        self.processing_start_time: Optional[float] = None
        
        # Threading for non-blocking operations
        self.progress_queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        
        # Enhanced statistics tracking
        self.processing_stats = {}
        
        # Language selection state
        self.language_vars = {}
        self.language_checkboxes = {}
        self.primary_lang_var = tk.StringVar()
        self.secondary_lang_var = tk.StringVar()
        
        # Get supported languages
        self.supported_languages = self.config_manager.get_supported_languages()
        self.language_list = list(self.supported_languages.items())
        
        # Build enhanced UI
        self._create_enhanced_widgets()
        self._setup_bindings()
        self._apply_ui_theme()
        
        # Start progress monitoring
        self._check_progress_queue()
        
        # Auto-save preferences periodically
        self._schedule_preference_save()
    
    def _load_ui_preferences(self):
        """Load UI preferences from config"""
        ui_config = self.config_manager.get_ui_config()
        
        # Apply window size
        self.root.geometry(f"{ui_config.window_width}x{ui_config.window_height}")
        
        # Load last used directories
        if ui_config.remember_directories:
            config_data = {}
            config_file = Path("gui_preferences.json")
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                except:
                    pass
            
            self.last_file_dir = config_data.get('last_file_dir', '')
            self.last_output_dir = config_data.get('last_output_dir', 'output_subtitles')
            self.last_audio_dir = config_data.get('last_audio_dir', 'boosted_audio')
        else:
            self.last_file_dir = ''
            self.last_output_dir = 'output_subtitles'
            self.last_audio_dir = 'boosted_audio'
    
    def _save_ui_preferences(self):
        """Save current UI preferences"""
        ui_config = self.config_manager.get_ui_config()
        
        if ui_config.remember_directories:
            config_data = {
                'last_file_dir': self.last_file_dir,
                'last_output_dir': getattr(self, 'subtitle_dir_var', tk.StringVar()).get(),
                'last_audio_dir': getattr(self, 'audio_dir_var', tk.StringVar()).get(),
                'window_geometry': self.root.geometry()
            }
            
            try:
                with open("gui_preferences.json", 'w') as f:
                    json.dump(config_data, f, indent=2)
            except:
                pass
    
    def _create_enhanced_widgets(self):
        """Create enhanced GUI widgets with better layout and features"""
        
        # Create main notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root, padding="5")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main processing tab
        self.main_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_frame, text="Processing")
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Build each tab
        self._create_main_tab()
        self._create_settings_tab()
        self._create_statistics_tab()
    
    def _create_main_tab(self):
        """Create the main processing tab"""
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        
        # File selection section
        self._create_enhanced_file_selection(self.main_frame, 0)
        
        # Processing options section
        self._create_enhanced_processing_options(self.main_frame, 1)
        
        # Output directories section
        self._create_enhanced_output_directories(self.main_frame, 2)
        
        # Language selection section
        self._create_enhanced_language_section(self.main_frame, 3)
        
        # Progress section with enhanced features
        self._create_enhanced_progress_section(self.main_frame, 4)
        
        # Control buttons with enhanced features
        self._create_enhanced_control_buttons(self.main_frame, 5)
        
        # Enhanced status display
        self._create_enhanced_status_section(self.main_frame, 6)
    
    def _create_enhanced_file_selection(self, parent, row):
        """Create enhanced file selection with drag-drop support"""
        section_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        section_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        section_frame.columnconfigure(1, weight=1)
        
        # File selection buttons
        button_frame = ttk.Frame(section_frame)
        button_frame.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(button_frame, text="Select Files", 
                  command=self._select_files).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(button_frame, text="Add Folder", 
                  command=self._select_folder).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", 
                  command=self._clear_files).pack(side=tk.LEFT, padx=5)
        
        # File list with scrollbar
        list_frame = ttk.Frame(section_frame)
        list_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(list_frame, height=4, selectmode=tk.EXTENDED)
        scrollbar_files = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, 
                                       command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar_files.set)
        
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_files.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # File validation indicator
        self.validation_var = tk.StringVar(value="No files selected")
        validation_label = ttk.Label(section_frame, textvariable=self.validation_var, 
                                   foreground="gray")
        validation_label.grid(row=2, column=0, columnspan=2, sticky=tk.W)
    
    def _create_enhanced_processing_options(self, parent, row):
        """Create enhanced processing options with advanced settings"""
        section_frame = ttk.LabelFrame(parent, text="Processing Options", padding="10")
        section_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        section_frame.columnconfigure(1, weight=1)
        
        # Audio processing options
        audio_frame = ttk.LabelFrame(section_frame, text="Audio Processing", padding="5")
        audio_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        self.boost_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(audio_frame, text="Boost Audio", 
                       variable=self.boost_audio_var,
                       command=self._toggle_boost_options).grid(row=0, column=0, sticky=tk.W)
        
        # Enhanced boost level with preview
        boost_frame = ttk.Frame(audio_frame)
        boost_frame.grid(row=1, column=0, sticky=tk.W, padx=(20, 0))
        
        ttk.Label(boost_frame, text="Boost Level:").grid(row=0, column=0, padx=(0, 5))
        
        self.boost_level_var = tk.IntVar(value=1)
        self.boost_scale = ttk.Scale(boost_frame, from_=1, to=10, 
                                    variable=self.boost_level_var, orient=tk.HORIZONTAL,
                                    length=200, command=self._update_boost_display)
        self.boost_scale.grid(row=0, column=1, padx=5)
        
        self.boost_level_label = ttk.Label(boost_frame, text="1x (No boost)")
        self.boost_level_label.grid(row=0, column=2, padx=5)
        
        # Transcription options
        transcription_frame = ttk.LabelFrame(section_frame, text="Transcription", padding="5")
        transcription_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(transcription_frame, text="Whisper Model:").grid(row=0, column=0, sticky=tk.W)
        
        self.model_var = tk.StringVar(value="medium")
        model_combo = ttk.Combobox(transcription_frame, textvariable=self.model_var,
                                  values=["tiny", "base", "small", "medium", "large"],
                                  state="readonly", width=15)
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Model info label
        self.model_info_var = tk.StringVar(value="Balanced speed and accuracy")
        ttk.Label(transcription_frame, textvariable=self.model_info_var, 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=0, column=2, padx=5)
        
        model_combo.bind('<<ComboboxSelected>>', self._update_model_info)
    
    def _create_enhanced_output_directories(self, parent, row):
        """Create enhanced output directories with validation"""
        section_frame = ttk.LabelFrame(parent, text="Output Directories", padding="10")
        section_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        section_frame.columnconfigure(1, weight=1)
        
        # Subtitles directory
        ttk.Label(section_frame, text="Subtitles:").grid(row=0, column=0, sticky=tk.W)
        self.subtitle_dir_var = tk.StringVar(value=self.last_output_dir)
        subtitle_entry = ttk.Entry(section_frame, textvariable=self.subtitle_dir_var)
        subtitle_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        button_frame1 = ttk.Frame(section_frame)
        button_frame1.grid(row=0, column=2)
        ttk.Button(button_frame1, text="Browse", width=8,
                  command=lambda: self._browse_directory(self.subtitle_dir_var)).pack(side=tk.LEFT)
        ttk.Button(button_frame1, text="Create", width=8,
                  command=lambda: self._create_directory(self.subtitle_dir_var)).pack(side=tk.LEFT, padx=(2, 0))
        
        # Boosted audio directory
        ttk.Label(section_frame, text="Boosted Audio:").grid(row=1, column=0, sticky=tk.W)
        self.audio_dir_var = tk.StringVar(value=self.last_audio_dir)
        audio_entry = ttk.Entry(section_frame, textvariable=self.audio_dir_var)
        audio_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        button_frame2 = ttk.Frame(section_frame)
        button_frame2.grid(row=1, column=2)
        ttk.Button(button_frame2, text="Browse", width=8,
                  command=lambda: self._browse_directory(self.audio_dir_var)).pack(side=tk.LEFT)
        ttk.Button(button_frame2, text="Create", width=8,
                  command=lambda: self._create_directory(self.audio_dir_var)).pack(side=tk.LEFT, padx=(2, 0))
        
        # Directory validation
        self.dir_validation_var = tk.StringVar()
        ttk.Label(section_frame, textvariable=self.dir_validation_var, 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=3, sticky=tk.W)
        
        # Bind validation
        self.subtitle_dir_var.trace_add('write', lambda *args: self._validate_directories())
        self.audio_dir_var.trace_add('write', lambda *args: self._validate_directories())
    
    def _create_enhanced_language_section(self, parent, row):
        """Create enhanced language selection with dual translation support"""
        section_frame = ttk.LabelFrame(parent, text="Translation Configuration", padding="10")
        section_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        section_frame.columnconfigure(1, weight=1)
        
        # Dual translation mode toggle
        dual_mode_frame = ttk.Frame(section_frame)
        dual_mode_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.dual_translation_var = tk.BooleanVar(
            value=self.config_manager.is_dual_translation_enabled()
        )
        dual_checkbox = ttk.Checkbutton(
            dual_mode_frame, 
            text="Enable Dual Language Translation", 
            variable=self.dual_translation_var,
            command=self._toggle_dual_translation_mode
        )
        dual_checkbox.pack(side=tk.LEFT)
        
        # Help label for dual translation
        help_label = ttk.Label(
            dual_mode_frame, 
            text="(Generate subtitles in two languages simultaneously)",
            foreground="gray", 
            font=("TkDefaultFont", 8)
        )
        help_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Language selection container that changes based on mode
        self.language_selection_frame = ttk.Frame(section_frame)
        self.language_selection_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Initialize language selection UI
        self._update_language_selection_ui()
        
        # Language preset buttons
        preset_frame = ttk.Frame(section_frame)
        preset_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        ttk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(preset_frame, text="None", command=lambda: self._apply_language_preset([])).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="Dual EN+ES", command=lambda: self._apply_dual_preset("en", "es")).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Dual EN+FR", command=lambda: self._apply_dual_preset("en", "fr")).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Dual EN+DE", command=lambda: self._apply_dual_preset("en", "de")).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="European", command=lambda: self._apply_language_preset(["es", "fr", "de", "it"])).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Asian", command=lambda: self._apply_language_preset(["zh", "ja", "ko"])).pack(side=tk.LEFT, padx=5)
        
        # Status display
        self.language_status_var = tk.StringVar()
        self._update_language_status()
        status_label = ttk.Label(section_frame, textvariable=self.language_status_var, 
                               foreground="blue", font=("TkDefaultFont", 9))
        status_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
    
    def _create_enhanced_progress_section(self, parent, row):
        """Create enhanced progress section with detailed information"""
        section_frame = ttk.LabelFrame(parent, text="Processing Progress", padding="10")
        section_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        section_frame.columnconfigure(0, weight=1)
        
        # Overall progress
        progress_info_frame = ttk.Frame(section_frame)
        progress_info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        progress_info_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_info_frame, variable=self.progress_var, 
                                          maximum=100, length=400, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Progress percentage label
        self.progress_percent_var = tk.StringVar(value="0%")
        ttk.Label(progress_info_frame, textvariable=self.progress_percent_var).grid(row=0, column=1, padx=5)
        
        # Status and timing information
        status_frame = ttk.Frame(section_frame)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky=tk.W)
        
        self.timing_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.timing_var, 
                 foreground="gray").grid(row=0, column=1, sticky=tk.E)
        
        # Processing statistics
        stats_frame = ttk.Frame(section_frame)
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.files_processed_var = tk.StringVar(value="Files: 0/0")
        ttk.Label(stats_frame, textvariable=self.files_processed_var).grid(row=0, column=0, sticky=tk.W)
        
        self.success_rate_var = tk.StringVar(value="Success: 0%")
        ttk.Label(stats_frame, textvariable=self.success_rate_var).grid(row=0, column=1, padx=20)
        
        self.speed_var = tk.StringVar(value="Speed: --")
        ttk.Label(stats_frame, textvariable=self.speed_var).grid(row=0, column=2, sticky=tk.E)
    
    def _create_enhanced_control_buttons(self, parent, row):
        """Create enhanced control buttons with additional options"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, pady=15)
        
        # Main processing button with enhanced state management
        self.process_button = ttk.Button(button_frame, text="Start Processing", 
                                       command=self._start_processing,
                                       style="Accent.TButton")
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        # Enhanced cancel button
        self.cancel_button = ttk.Button(button_frame, text="Cancel", 
                                      command=self._cancel_processing, 
                                      state="disabled")
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Pause/Resume button (future enhancement)
        self.pause_button = ttk.Button(button_frame, text="Pause", 
                                     command=self._pause_processing, 
                                     state="disabled")
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Clear results button
        ttk.Button(button_frame, text="Clear All", 
                  command=self._clear_all).pack(side=tk.LEFT, padx=5)
        
        # Settings shortcut
        ttk.Button(button_frame, text="Settings", 
                  command=lambda: self.notebook.select(1)).pack(side=tk.LEFT, padx=5)
        
        # Open output folder
        self.open_output_button = ttk.Button(button_frame, text="Open Output", 
                                           command=self._open_output_folder)
        self.open_output_button.pack(side=tk.LEFT, padx=5)
    
    def _create_enhanced_status_section(self, parent, row):
        """Create enhanced status section with filtering and export"""
        section_frame = ttk.LabelFrame(parent, text="Processing Log", padding="10")
        section_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        section_frame.columnconfigure(0, weight=1)
        section_frame.rowconfigure(1, weight=1)
        
        # Log controls
        log_controls = ttk.Frame(section_frame)
        log_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(log_controls, text="Clear Log", 
                  command=self._clear_status_log).pack(side=tk.LEFT)
        ttk.Button(log_controls, text="Save Log", 
                  command=self._save_status_log).pack(side=tk.LEFT, padx=5)
        
        # Log level filter
        ttk.Label(log_controls, text="Level:").pack(side=tk.LEFT, padx=(20, 5))
        self.log_level_var = tk.StringVar(value="All")
        log_level_combo = ttk.Combobox(log_controls, textvariable=self.log_level_var,
                                      values=["All", "Info", "Warning", "Error"],
                                      state="readonly", width=8)
        log_level_combo.pack(side=tk.LEFT)
        
        # Enhanced status text area with better formatting
        text_frame = ttk.Frame(section_frame)
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.status_text = tk.Text(text_frame, height=10, wrap=tk.WORD, 
                                  font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, 
                                 command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure text tags for different log levels
        self.status_text.tag_configure("info", foreground="black")
        self.status_text.tag_configure("warning", foreground="orange")
        self.status_text.tag_configure("error", foreground="red")
        self.status_text.tag_configure("success", foreground="green")
    
    def _create_settings_tab(self):
        """Create the settings configuration tab"""
        # Settings will be added here
        ttk.Label(self.settings_frame, text="Settings configuration coming soon...", 
                 font=("TkDefaultFont", 12)).pack(pady=50)
    
    def _create_statistics_tab(self):
        """Create the statistics and monitoring tab"""
        # Statistics will be added here  
        ttk.Label(self.stats_frame, text="Processing statistics coming soon...", 
                 font=("TkDefaultFont", 12)).pack(pady=50)
    
    # Dual language translation methods
    def _toggle_dual_translation_mode(self):
        """Toggle between single and dual language translation modes."""
        enabled = self.dual_translation_var.get()
        
        # Update configuration
        primary_lang, secondary_lang = self.config_manager.get_translation_languages()
        self.config_manager.set_dual_translation(enabled, primary_lang, secondary_lang)
        
        # Update UI
        self._update_language_selection_ui()
        self._update_language_status()
        
        # Log the change
        mode_text = "enabled" if enabled else "disabled"
        self._log_message(f"Dual language translation {mode_text}", "info")
    
    def _update_language_selection_ui(self):
        """Update the language selection UI based on current mode."""
        # Clear existing widgets
        for widget in self.language_selection_frame.winfo_children():
            widget.destroy()
        
        if self.dual_translation_var.get():
            self._create_dual_language_selection()
        else:
            self._create_multi_language_selection()
    
    def _create_dual_language_selection(self):
        """Create dual language selection interface."""
        # Primary language selection
        primary_frame = ttk.Frame(self.language_selection_frame)
        primary_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(primary_frame, text="Primary Language:").pack(side=tk.LEFT, padx=(0, 10))
        
        primary_lang, secondary_lang = self.config_manager.get_translation_languages()
        self.primary_lang_var.set(primary_lang)
        
        primary_combo = ttk.Combobox(
            primary_frame,
            textvariable=self.primary_lang_var,
            values=[name for name, code in self.language_list if code != "auto"],
            state="readonly",
            width=15
        )
        primary_combo.pack(side=tk.LEFT, padx=5)
        primary_combo.bind('<<ComboboxSelected>>', self._on_primary_language_changed)
        
        # Secondary language selection
        secondary_frame = ttk.Frame(self.language_selection_frame)
        secondary_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(secondary_frame, text="Secondary Language:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.secondary_lang_var.set(secondary_lang)
        
        secondary_combo = ttk.Combobox(
            secondary_frame,
            textvariable=self.secondary_lang_var,
            values=[name for name, code in self.language_list if code != "auto"],
            state="readonly",
            width=15
        )
        secondary_combo.pack(side=tk.LEFT, padx=5)
        secondary_combo.bind('<<ComboboxSelected>>', self._on_secondary_language_changed)
        
        # Swap button
        swap_button = ttk.Button(
            secondary_frame,
            text="⇅ Swap",
            command=self._swap_languages,
            width=8
        )
        swap_button.pack(side=tk.LEFT, padx=10)
    
    def _create_multi_language_selection(self):
        """Create multi-language checkbox selection interface."""
        # Language checkboxes with better organization
        languages_frame = ttk.Frame(self.language_selection_frame)
        languages_frame.pack(fill=tk.X, pady=5)
        
        self.language_vars = {}
        self.language_checkboxes = {}
        
        languages = [
            ("Spanish", "es"), ("French", "fr"), ("German", "de"), ("Italian", "it"),
            ("Portuguese", "pt"), ("Russian", "ru"), ("Chinese", "zh"), ("Japanese", "ja"),
            ("Korean", "ko"), ("Arabic", "ar"), ("Hindi", "hi"), ("Dutch", "nl"),
            ("Swedish", "sv"), ("Norwegian", "no"), ("Danish", "da"), ("Finnish", "fi"),
            ("Polish", "pl"), ("Ukrainian", "uk")
        ]
        
        # Create checkboxes in a grid (4 columns)
        for i, (lang_name, lang_code) in enumerate(languages):
            var = tk.BooleanVar()
            self.language_vars[lang_code] = var
            
            row_pos = i // 4
            col_pos = i % 4
            
            cb = ttk.Checkbutton(languages_frame, text=lang_name, variable=var,
                                command=self._update_language_status)
            cb.grid(row=row_pos, column=col_pos, sticky=tk.W, padx=10, pady=1)
            self.language_checkboxes[lang_code] = cb
    
    def _on_primary_language_changed(self, event=None):
        """Handle primary language selection change."""
        selected_name = self.primary_lang_var.get()
        selected_code = next((code for name, code in self.language_list if name == selected_name), "en")
        
        # Update configuration
        _, secondary_lang = self.config_manager.get_translation_languages()
        self.config_manager.set_dual_translation(True, selected_code, secondary_lang)
        
        self._update_language_status()
    
    def _on_secondary_language_changed(self, event=None):
        """Handle secondary language selection change."""
        selected_name = self.secondary_lang_var.get()
        selected_code = next((code for name, code in self.language_list if name == selected_name), "es")
        
        # Update configuration
        primary_lang, _ = self.config_manager.get_translation_languages()
        self.config_manager.set_dual_translation(True, primary_lang, selected_code)
        
        self._update_language_status()
    
    def _swap_languages(self):
        """Swap primary and secondary languages."""
        primary_name = self.primary_lang_var.get()
        secondary_name = self.secondary_lang_var.get()
        
        self.primary_lang_var.set(secondary_name)
        self.secondary_lang_var.set(primary_name)
        
        # Trigger the change events
        self._on_primary_language_changed()
        self._on_secondary_language_changed()
    
    def _apply_dual_preset(self, primary_lang: str, secondary_lang: str):
        """Apply a dual language preset."""
        self.dual_translation_var.set(True)
        self.config_manager.set_dual_translation(True, primary_lang, secondary_lang)
        
        # Update UI
        self._update_language_selection_ui()
        self._update_language_status()
        
        self._log_message(f"Applied dual language preset: {primary_lang} + {secondary_lang}", "info")
    
    def _apply_language_preset(self, lang_codes: list):
        """Apply a multi-language preset."""
        if self.dual_translation_var.get():
            # If in dual mode, disable it
            self.dual_translation_var.set(False)
            self.config_manager.set_dual_translation(False)
            self._update_language_selection_ui()
        
        # Clear all checkboxes first
        for var in self.language_vars.values():
            var.set(False)
        
        # Set selected languages
        for lang_code in lang_codes:
            if lang_code in self.language_vars:
                self.language_vars[lang_code].set(True)
        
        self._update_language_status()
        
        if lang_codes:
            self._log_message(f"Applied language preset: {', '.join(lang_codes)}", "info")
        else:
            self._log_message("Cleared all language selections", "info")
    
    def _update_language_status(self):
        """Update the language selection status display."""
        if self.dual_translation_var.get():
            primary_name = self.primary_lang_var.get() or "English"
            secondary_name = self.secondary_lang_var.get() or "Spanish"
            status_text = f"Dual translation: {primary_name} + {secondary_name}"
        else:
            selected_count = sum(1 for var in self.language_vars.values() if var.get())
            if selected_count == 0:
                status_text = "No languages selected for translation"
            elif selected_count == 1:
                selected_langs = [code for code, var in self.language_vars.items() if var.get()]
                lang_name = next((name for name, code in self.language_list if code == selected_langs[0]), selected_langs[0])
                status_text = f"1 language selected: {lang_name}"
            else:
                status_text = f"{selected_count} languages selected for translation"
        
        self.language_status_var.set(status_text)
    
    def _log_message(self, message: str, level: str = "info"):
        """Add a message to the status log."""
        # This is a placeholder - implement actual logging to status text widget
        print(f"[{level.upper()}] {message}")
    
    # File selection and utility methods
    def _select_files(self):
        """Select input files for processing."""
        filetypes = [
            ("Media Files", "*.mp4 *.mp3 *.wav *.m4a"),
            ("MP4 Files", "*.mp4"),
            ("Audio Files", "*.mp3 *.wav *.m4a"),
            ("All Files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Media Files",
            initialdir=self.last_file_dir,
            filetypes=filetypes
        )
        
        if files:
            self.selected_files.extend(files)
            self._update_file_list()
            # Update last directory
            self.last_file_dir = str(Path(files[0]).parent)
    
    def _select_folder(self):
        """Select a folder and add all media files from it."""
        folder = filedialog.askdirectory(
            title="Select Folder with Media Files",
            initialdir=self.last_file_dir
        )
        
        if folder:
            # Find all media files in the folder
            media_extensions = ('.mp4', '.mp3', '.wav', '.m4a')
            folder_path = Path(folder)
            
            media_files = []
            for ext in media_extensions:
                media_files.extend(folder_path.glob(f"*{ext}"))
                media_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            if media_files:
                self.selected_files.extend([str(f) for f in media_files])
                self._update_file_list()
                self.last_file_dir = folder
                self._log_message(f"Added {len(media_files)} files from folder: {folder}", "info")
            else:
                messagebox.showwarning("No Files", "No media files found in the selected folder.")
    
    def _clear_files(self):
        """Clear the selected files list."""
        self.selected_files.clear()
        self._update_file_list()
        self._log_message("File list cleared", "info")
    
    def _update_file_list(self):
        """Update the file list display."""
        self.file_listbox.delete(0, tk.END)
        
        for file_path in self.selected_files:
            file_name = Path(file_path).name
            self.file_listbox.insert(tk.END, file_name)
        
        # Update validation status
        file_count = len(self.selected_files)
        if file_count == 0:
            self.validation_var.set("No files selected")
        else:
            self.validation_var.set(f"{file_count} file(s) selected")
    
    # Placeholder methods for missing functionality
    def _start_processing(self):
        """Start the processing workflow."""
        # This would implement the actual processing logic
        pass
    
    def _cancel_processing(self):
        """Cancel the current processing."""
        pass
    
    def _pause_processing(self):
        """Pause/resume processing."""
        pass
    
    def _clear_all(self):
        """Clear all data and reset UI."""
        self._clear_files()
        self._apply_language_preset([])
    
    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        output_dir = self.subtitle_dir_var.get()
        if Path(output_dir).exists():
            os.startfile(output_dir)
    
    def _browse_directory(self, dir_var):
        """Browse for a directory."""
        directory = filedialog.askdirectory(
            title="Select Directory",
            initialdir=dir_var.get()
        )
        if directory:
            dir_var.set(directory)
    
    def _create_directory(self, dir_var):
        """Create the specified directory."""
        directory = dir_var.get()
        if directory:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                self._log_message(f"Created directory: {directory}", "success")
            except Exception as e:
                self._log_message(f"Failed to create directory: {e}", "error")
    
    def _validate_directories(self):
        """Validate output directories."""
        # Placeholder for directory validation
        pass
    
    def _toggle_boost_options(self):
        """Toggle audio boost options."""
        # Enable/disable boost controls based on checkbox
        pass
    
    def _update_boost_display(self, value):
        """Update boost level display."""
        level = int(float(value))
        if level == 1:
            self.boost_level_label.config(text="1x (No boost)")
        else:
            self.boost_level_label.config(text=f"{level}x")
    
    def _update_model_info(self, event=None):
        """Update Whisper model information."""
        model = self.model_var.get()
        info_map = {
            "tiny": "Fastest, lowest accuracy",
            "base": "Fast, basic accuracy",
            "small": "Good speed/accuracy balance",
            "medium": "Balanced speed and accuracy",
            "large": "Highest accuracy, slower"
        }
        self.model_info_var.set(info_map.get(model, "Unknown model"))
    
    def _show_file_context_menu(self, event):
        """Show context menu for file list."""
        # Placeholder for context menu
        pass
    
    def _clear_status_log(self):
        """Clear the status log."""
        self.status_text.delete(1.0, tk.END)
    
    def _save_status_log(self):
        """Save the status log to file."""
        # Placeholder for log saving
        pass
    
    def _update_progress(self, progress: float, message: str):
        """Update progress display."""
        self.progress_var.set(progress * 100)
        self.progress_percent_var.set(f"{progress * 100:.1f}%")
        self.status_var.set(message)
    
    def _check_progress_queue(self):
        """Check for progress updates in the queue."""
        # Placeholder for progress monitoring
        self.root.after(100, self._check_progress_queue)
    
    def _on_closing(self):
        """Handle application closing."""
        self._save_ui_preferences()
        self.root.destroy()
    
    def _setup_bindings(self):
        """Setup enhanced event bindings"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Bind file listbox for context menu
        self.file_listbox.bind("<Button-3>", self._show_file_context_menu)
        
        # Auto-validation bindings
        self.subtitle_dir_var.trace_add('write', lambda *args: self._validate_directories())
        self.audio_dir_var.trace_add('write', lambda *args: self._validate_directories())
        
        # Initialize language selection based on current config
        if hasattr(self, 'dual_translation_var'):
            self._update_language_selection_ui()
    
    def _apply_ui_theme(self):
        """Apply UI theme and styling"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure("Accent.TButton", foreground="white", 
                       font=("TkDefaultFont", 9, "bold"))
    
    def _schedule_preference_save(self):
        """Schedule periodic preference saving"""
        self._save_ui_preferences()
        self.root.after(30000, self._schedule_preference_save)  # Save every 30 seconds
    
    def run(self):
        """Start the enhanced GUI application"""
        self.root.mainloop()


def main():
    """Enhanced application entry point"""
    app = EnhancedTranscriptionGUI()
    app.run()


if __name__ == "__main__":
    main()