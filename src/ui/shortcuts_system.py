"""
Custom Shortcuts System
Provides configurable keyboard shortcuts for common operations.
"""

import json
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
        QLineEdit, QComboBox, QMessageBox, QGroupBox, QCheckBox,
        QTabWidget, QTextEdit, QSplitter, QFrame
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSettings
    from PyQt6.QtGui import (
        QKeySequence, QShortcut, QAction, QFont, QIcon, 
        QKeyEvent, QPalette, QColor
    )
    HAS_QT = True
except ImportError:
    HAS_QT = False

logger = logging.getLogger(__name__)

class ActionCategory(Enum):
    """Categories for shortcut actions"""
    FILE = "File Operations"
    PROCESSING = "Processing"
    PLAYBACK = "Playback Control"
    VIEW = "View & Navigation"
    EDITING = "Text Editing"
    WORKFLOW = "Workflow"

@dataclass
class ShortcutAction:
    """Represents a shortcut action"""
    id: str
    name: str
    description: str
    category: ActionCategory
    default_shortcut: str
    current_shortcut: str = ""
    callback: Optional[Callable] = None
    context: str = "global"  # global, transcription, editing, etc.
    enabled: bool = True
    icon: Optional[str] = None
    
    def __post_init__(self):
        if not self.current_shortcut:
            self.current_shortcut = self.default_shortcut

class ShortcutManager:
    """Manages keyboard shortcuts and actions"""
    
    DEFAULT_SHORTCUTS = {
        # File Operations
        "open_files": ShortcutAction(
            "open_files", "Open Files", "Open audio/video files for processing",
            ActionCategory.FILE, "Ctrl+O"
        ),
        "save_transcription": ShortcutAction(
            "save_transcription", "Save Transcription", "Save current transcription",
            ActionCategory.FILE, "Ctrl+S"
        ),
        "export_subtitles": ShortcutAction(
            "export_subtitles", "Export Subtitles", "Export subtitles to SRT",
            ActionCategory.FILE, "Ctrl+E"
        ),
        "clear_all": ShortcutAction(
            "clear_all", "Clear All", "Clear all files and transcriptions",
            ActionCategory.FILE, "Ctrl+Shift+C"
        ),
        
        # Processing
        "start_processing": ShortcutAction(
            "start_processing", "Start Processing", "Begin transcription processing",
            ActionCategory.PROCESSING, "F5"
        ),
        "stop_processing": ShortcutAction(
            "stop_processing", "Stop Processing", "Stop current processing",
            ActionCategory.PROCESSING, "F6"
        ),
        "pause_processing": ShortcutAction(
            "pause_processing", "Pause Processing", "Pause/resume processing",
            ActionCategory.PROCESSING, "F7"
        ),
        "boost_audio": ShortcutAction(
            "boost_audio", "Boost Audio", "Apply audio volume boost",
            ActionCategory.PROCESSING, "Ctrl+B"
        ),
        "translate_text": ShortcutAction(
            "translate_text", "Translate", "Translate transcription",
            ActionCategory.PROCESSING, "Ctrl+T"
        ),
        
        # Playback Control
        "play_pause": ShortcutAction(
            "play_pause", "Play/Pause", "Toggle audio playback",
            ActionCategory.PLAYBACK, "Space", context="playback"
        ),
        "seek_backward": ShortcutAction(
            "seek_backward", "Seek Backward", "Seek backward 10 seconds",
            ActionCategory.PLAYBACK, "Left", context="playback"
        ),
        "seek_forward": ShortcutAction(
            "seek_forward", "Seek Forward", "Seek forward 10 seconds",
            ActionCategory.PLAYBACK, "Right", context="playback"
        ),
        "volume_up": ShortcutAction(
            "volume_up", "Volume Up", "Increase playback volume",
            ActionCategory.PLAYBACK, "Up", context="playback"
        ),
        "volume_down": ShortcutAction(
            "volume_down", "Volume Down", "Decrease playback volume",
            ActionCategory.PLAYBACK, "Down", context="playback"
        ),
        
        # View & Navigation
        "toggle_preview": ShortcutAction(
            "toggle_preview", "Toggle Preview", "Show/hide live preview",
            ActionCategory.VIEW, "F1"
        ),
        "toggle_waveform": ShortcutAction(
            "toggle_waveform", "Toggle Waveform", "Show/hide audio waveform",
            ActionCategory.VIEW, "F2"
        ),
        "zoom_in": ShortcutAction(
            "zoom_in", "Zoom In", "Zoom in timeline/waveform",
            ActionCategory.VIEW, "Ctrl+Plus"
        ),
        "zoom_out": ShortcutAction(
            "zoom_out", "Zoom Out", "Zoom out timeline/waveform",
            ActionCategory.VIEW, "Ctrl+Minus"
        ),
        "fit_to_window": ShortcutAction(
            "fit_to_window", "Fit to Window", "Fit content to window",
            ActionCategory.VIEW, "Ctrl+0"
        ),
        
        # Text Editing
        "find_text": ShortcutAction(
            "find_text", "Find Text", "Find text in transcription",
            ActionCategory.EDITING, "Ctrl+F", context="editing"
        ),
        "replace_text": ShortcutAction(
            "replace_text", "Replace Text", "Find and replace text",
            ActionCategory.EDITING, "Ctrl+H", context="editing"
        ),
        "select_all": ShortcutAction(
            "select_all", "Select All", "Select all transcription text",
            ActionCategory.EDITING, "Ctrl+A", context="editing"
        ),
        "undo": ShortcutAction(
            "undo", "Undo", "Undo last action",
            ActionCategory.EDITING, "Ctrl+Z", context="editing"
        ),
        "redo": ShortcutAction(
            "redo", "Redo", "Redo last undone action",
            ActionCategory.EDITING, "Ctrl+Y", context="editing"
        ),
        
        # Workflow
        "quick_transcribe": ShortcutAction(
            "quick_transcribe", "Quick Transcribe", "Transcribe selected files with defaults",
            ActionCategory.WORKFLOW, "Ctrl+Q"
        ),
        "batch_process": ShortcutAction(
            "batch_process", "Batch Process", "Start batch processing",
            ActionCategory.WORKFLOW, "Ctrl+Shift+P"
        ),
        "settings": ShortcutAction(
            "settings", "Settings", "Open application settings",
            ActionCategory.WORKFLOW, "Ctrl+,"
        ),
        "help": ShortcutAction(
            "help", "Help", "Show help documentation",
            ActionCategory.WORKFLOW, "F11"
        ),
    }
    
    def __init__(self, config_dir: Path = None):
        if not HAS_QT:
            raise ImportError("PyQt6 required for shortcuts system")
            
        self.config_dir = config_dir or Path.home() / ".config" / "transcription_tool"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.shortcuts_file = self.config_dir / "shortcuts.json"
        
        self.actions: Dict[str, ShortcutAction] = {}
        self.shortcuts: Dict[str, QShortcut] = {}
        self.contexts: Dict[str, QWidget] = {}
        
        self.load_shortcuts()
        
    def load_shortcuts(self):
        """Load shortcuts from configuration"""
        # Start with defaults
        self.actions = self.DEFAULT_SHORTCUTS.copy()
        
        # Load custom shortcuts if they exist
        if self.shortcuts_file.exists():
            try:
                with open(self.shortcuts_file, 'r') as f:
                    data = json.load(f)
                    
                for action_id, shortcut_data in data.items():
                    if action_id in self.actions:
                        action = self.actions[action_id]
                        action.current_shortcut = shortcut_data.get('shortcut', action.default_shortcut)
                        action.enabled = shortcut_data.get('enabled', True)
                        
                logger.info(f"Loaded {len(data)} custom shortcuts")
                        
            except Exception as e:
                logger.error(f"Failed to load shortcuts: {e}")
                
    def save_shortcuts(self):
        """Save shortcuts to configuration"""
        try:
            data = {}
            for action_id, action in self.actions.items():
                if action.current_shortcut != action.default_shortcut or not action.enabled:
                    data[action_id] = {
                        'shortcut': action.current_shortcut,
                        'enabled': action.enabled
                    }
                    
            with open(self.shortcuts_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(data)} custom shortcuts")
            
        except Exception as e:
            logger.error(f"Failed to save shortcuts: {e}")
            
    def register_context(self, context_name: str, widget: QWidget):
        """Register a context widget for shortcuts"""
        self.contexts[context_name] = widget
        logger.info(f"Registered shortcut context: {context_name}")
        
    def setup_shortcuts(self, parent_widget: QWidget):
        """Setup all shortcuts for the parent widget"""
        self.clear_shortcuts()
        
        for action_id, action in self.actions.items():
            if action.enabled and action.current_shortcut:
                self.create_shortcut(action_id, action, parent_widget)
                
        logger.info(f"Setup {len(self.shortcuts)} shortcuts")
        
    def create_shortcut(self, action_id: str, action: ShortcutAction, parent: QWidget):
        """Create a single shortcut"""
        try:
            # Get context widget
            context_widget = self.contexts.get(action.context, parent)
            
            # Create shortcut
            shortcut = QShortcut(QKeySequence(action.current_shortcut), context_widget)
            
            # Connect to callback if available
            if action.callback:
                shortcut.activated.connect(action.callback)
            else:
                # Default callback that emits a signal
                shortcut.activated.connect(lambda aid=action_id: self.action_triggered(aid))
                
            self.shortcuts[action_id] = shortcut
            
        except Exception as e:
            logger.error(f"Failed to create shortcut for {action_id}: {e}")
            
    def action_triggered(self, action_id: str):
        """Handle action trigger when no specific callback is set"""
        logger.info(f"Shortcut action triggered: {action_id}")
        # This can be connected to by external code
        
    def clear_shortcuts(self):
        """Clear all current shortcuts"""
        for shortcut in self.shortcuts.values():
            shortcut.deleteLater()
        self.shortcuts.clear()
        
    def set_action_callback(self, action_id: str, callback: Callable):
        """Set callback for an action"""
        if action_id in self.actions:
            self.actions[action_id].callback = callback
            
            # Update existing shortcut
            if action_id in self.shortcuts:
                self.shortcuts[action_id].activated.disconnect()
                self.shortcuts[action_id].activated.connect(callback)
                
    def enable_action(self, action_id: str, enabled: bool):
        """Enable/disable an action"""
        if action_id in self.actions:
            self.actions[action_id].enabled = enabled
            
            if action_id in self.shortcuts:
                self.shortcuts[action_id].setEnabled(enabled)
                
    def set_shortcut(self, action_id: str, new_shortcut: str) -> bool:
        """Set new shortcut for action"""
        if action_id not in self.actions:
            return False
            
        # Check for conflicts
        conflict = self.find_shortcut_conflict(new_shortcut, action_id)
        if conflict:
            logger.warning(f"Shortcut conflict: {new_shortcut} already used by {conflict}")
            return False
            
        # Update action
        action = self.actions[action_id]
        old_shortcut = action.current_shortcut
        action.current_shortcut = new_shortcut
        
        # Update actual shortcut
        if action_id in self.shortcuts:
            try:
                self.shortcuts[action_id].setKey(QKeySequence(new_shortcut))
                logger.info(f"Updated shortcut for {action_id}: {old_shortcut} -> {new_shortcut}")
                return True
            except Exception as e:
                # Revert on error
                action.current_shortcut = old_shortcut
                logger.error(f"Failed to update shortcut: {e}")
                return False
                
        return True
        
    def find_shortcut_conflict(self, shortcut: str, exclude_action: str = None) -> Optional[str]:
        """Find if shortcut conflicts with existing one"""
        for action_id, action in self.actions.items():
            if action_id != exclude_action and action.current_shortcut == shortcut and action.enabled:
                return action_id
        return None
        
    def reset_to_defaults(self):
        """Reset all shortcuts to defaults"""
        for action in self.actions.values():
            action.current_shortcut = action.default_shortcut
            action.enabled = True
            
        # Remove custom config file
        if self.shortcuts_file.exists():
            self.shortcuts_file.unlink()
            
        logger.info("Reset all shortcuts to defaults")
        
    def get_actions_by_category(self) -> Dict[ActionCategory, List[ShortcutAction]]:
        """Get actions grouped by category"""
        categories = {}
        for action in self.actions.values():
            if action.category not in categories:
                categories[action.category] = []
            categories[action.category].append(action)
            
        return categories
        
    def export_shortcuts(self, file_path: Path):
        """Export shortcuts to file"""
        try:
            data = {
                'version': '1.0',
                'shortcuts': {}
            }
            
            for action_id, action in self.actions.items():
                data['shortcuts'][action_id] = {
                    'name': action.name,
                    'shortcut': action.current_shortcut,
                    'default': action.default_shortcut,
                    'enabled': action.enabled,
                    'category': action.category.value,
                    'description': action.description
                }
                
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Exported shortcuts to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export shortcuts: {e}")
            return False
            
    def import_shortcuts(self, file_path: Path) -> bool:
        """Import shortcuts from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            shortcuts_data = data.get('shortcuts', {})
            imported_count = 0
            
            for action_id, shortcut_data in shortcuts_data.items():
                if action_id in self.actions:
                    action = self.actions[action_id]
                    new_shortcut = shortcut_data.get('shortcut', '')
                    
                    # Check for conflicts
                    if new_shortcut and not self.find_shortcut_conflict(new_shortcut, action_id):
                        action.current_shortcut = new_shortcut
                        action.enabled = shortcut_data.get('enabled', True)
                        imported_count += 1
                        
            logger.info(f"Imported {imported_count} shortcuts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import shortcuts: {e}")
            return False

class ShortcutConfigDialog(QDialog):
    """Dialog for configuring shortcuts"""
    
    def __init__(self, shortcut_manager: ShortcutManager, parent=None):
        if not HAS_QT:
            raise ImportError("PyQt6 required for shortcut configuration")
            
        super().__init__(parent)
        self.shortcut_manager = shortcut_manager
        self.setWindowTitle("Keyboard Shortcuts Configuration")
        self.setModal(True)
        self.resize(800, 600)
        
        self.setup_ui()
        self.load_shortcuts()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs for each category
        categories = self.shortcut_manager.get_actions_by_category()
        for category, actions in categories.items():
            tab_widget = self.create_category_tab(category, actions)
            self.tabs.addTab(tab_widget, category.value)
            
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export...")
        self.export_btn.clicked.connect(self.export_shortcuts)
        
        self.import_btn = QPushButton("Import...")
        self.import_btn.clicked.connect(self.import_shortcuts)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_defaults)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDefault(True)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.import_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addWidget(self.tabs)
        layout.addLayout(button_layout)
        
    def create_category_tab(self, category: ActionCategory, actions: List[ShortcutAction]) -> QWidget:
        """Create tab for category"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create table
        table = QTableWidget(len(actions), 4)
        table.setHorizontalHeaderLabels(["Action", "Description", "Shortcut", "Enabled"])
        
        # Configure table
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        # Populate table
        for row, action in enumerate(actions):
            # Action name
            name_item = QTableWidgetItem(action.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name_item)
            
            # Description
            desc_item = QTableWidgetItem(action.description)
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 1, desc_item)
            
            # Shortcut
            shortcut_item = QTableWidgetItem(action.current_shortcut)
            table.setItem(row, 2, shortcut_item)
            
            # Enabled checkbox
            enabled_item = QTableWidgetItem()
            enabled_item.setCheckState(Qt.CheckState.Checked if action.enabled else Qt.CheckState.Unchecked)
            enabled_item.setFlags(enabled_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 3, enabled_item)
            
        layout.addWidget(table)
        
        # Store table reference for later use
        setattr(widget, '_table', table)
        setattr(widget, '_actions', actions)
        
        return widget
        
    def load_shortcuts(self):
        """Load current shortcuts into tables"""
        # Already done in create_category_tab
        pass
        
    def accept(self):
        """Save changes and close"""
        # Collect changes from all tabs
        for i in range(self.tabs.count()):
            tab_widget = self.tabs.widget(i)
            table = getattr(tab_widget, '_table')
            actions = getattr(tab_widget, '_actions')
            
            for row, action in enumerate(actions):
                # Get shortcut
                shortcut_item = table.item(row, 2)
                new_shortcut = shortcut_item.text().strip()
                
                # Get enabled state
                enabled_item = table.item(row, 3)
                enabled = enabled_item.checkState() == Qt.CheckState.Checked
                
                # Update action
                if new_shortcut != action.current_shortcut:
                    conflict = self.shortcut_manager.find_shortcut_conflict(new_shortcut, action.id)
                    if conflict and new_shortcut:
                        # Show conflict warning
                        conflict_action = self.shortcut_manager.actions[conflict]
                        reply = QMessageBox.question(
                            self,
                            "Shortcut Conflict",
                            f"Shortcut '{new_shortcut}' is already used by '{conflict_action.name}'.\n"
                            "Do you want to remove it from that action and assign it to this one?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        
                        if reply == QMessageBox.StandardButton.Yes:
                            # Remove from conflicting action
                            conflict_action.current_shortcut = ""
                            action.current_shortcut = new_shortcut
                        # else keep original shortcut
                    else:
                        action.current_shortcut = new_shortcut
                        
                action.enabled = enabled
                
        # Save to file
        self.shortcut_manager.save_shortcuts()
        
        super().accept()
        
    def reset_defaults(self):
        """Reset all shortcuts to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset Shortcuts",
            "Are you sure you want to reset all shortcuts to their defaults?\n"
            "This will lose all your customizations.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.shortcut_manager.reset_to_defaults()
            # Reload dialog
            self.reject()  # Close dialog
            # Parent should reopen it
            
    def export_shortcuts(self):
        """Export shortcuts to file"""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Shortcuts",
            str(Path.home() / "shortcuts.json"),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            success = self.shortcut_manager.export_shortcuts(Path(file_path))
            if success:
                QMessageBox.information(self, "Success", "Shortcuts exported successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to export shortcuts.")
                
    def import_shortcuts(self):
        """Import shortcuts from file"""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Shortcuts",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            success = self.shortcut_manager.import_shortcuts(Path(file_path))
            if success:
                QMessageBox.information(self, "Success", "Shortcuts imported successfully!")
                self.reject()  # Close and let parent reload
            else:
                QMessageBox.warning(self, "Error", "Failed to import shortcuts.")

# Global shortcut manager instance
shortcut_manager = ShortcutManager()