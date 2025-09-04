"""
Drag & Drop Interface for GUI
Provides modern drag & drop functionality for file handling.
"""

import os
from typing import List, Callable, Optional
from pathlib import Path
import logging

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
        QFrame, QFileDialog, QScrollArea, QListWidget, QListWidgetItem,
        QProgressBar, QApplication, QSizePolicy
    )
    from PyQt6.QtCore import Qt, QMimeData, pyqtSignal, QSize, QTimer
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QPen, QFont, QPixmap, QIcon
    HAS_QT = True
except ImportError:
    try:
        from tkinter import ttk, filedialog
        import tkinterdnd2 as tkdnd
        HAS_TK_DND = True
    except ImportError:
        HAS_TK_DND = False
    HAS_QT = False

logger = logging.getLogger(__name__)

class FileDropWidget(QWidget if HAS_QT else object):
    """Drag & drop widget for file handling"""
    
    # Signals for file operations
    files_dropped = pyqtSignal(list) if HAS_QT else None
    files_selected = pyqtSignal(list) if HAS_QT else None
    file_removed = pyqtSignal(str) if HAS_QT else None
    clear_all = pyqtSignal() if HAS_QT else None
    
    SUPPORTED_EXTENSIONS = {
        '.mp3', '.mp4', '.wav', '.flac', '.aac', '.ogg', '.m4a', 
        '.avi', '.mov', '.mkv', '.webm', '.wmv'
    }
    
    def __init__(self, parent=None):
        if not HAS_QT:
            raise ImportError("PyQt6 required for drag & drop interface")
            
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.dropped_files: List[str] = []
        self.file_callbacks: List[Callable[[List[str]], None]] = []
        
        self.setup_ui()
        self.setup_styling()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        # Drop zone
        self.drop_zone = QFrame()
        self.drop_zone.setFrameStyle(QFrame.Shape.StyledPanel)
        self.drop_zone.setLineWidth(2)
        self.drop_zone.setMidLineWidth(1)
        self.drop_zone.setMinimumHeight(150)
        self.drop_zone.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Drop zone layout
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.setSpacing(10)
        
        # Drop icon/text
        self.drop_label = QLabel("🎵 Drag & Drop Audio/Video Files Here\n\nor click Browse to select files")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setWordWrap(True)
        
        # Browse button
        self.browse_btn = QPushButton("📁 Browse Files")
        self.browse_btn.setMaximumWidth(150)
        self.browse_btn.clicked.connect(self.browse_files)
        
        drop_layout.addWidget(self.drop_label)
        drop_layout.addWidget(self.browse_btn)
        
        # File list area
        self.file_list_frame = QFrame()
        list_layout = QVBoxLayout(self.file_list_frame)
        list_layout.setContentsMargins(0, 0, 0, 0)
        
        # File list header
        list_header = QHBoxLayout()
        self.file_count_label = QLabel("Selected Files (0)")
        self.file_count_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        self.clear_all_btn = QPushButton("🗑️ Clear All")
        self.clear_all_btn.setMaximumWidth(100)
        self.clear_all_btn.clicked.connect(self.clear_all_files)
        self.clear_all_btn.setVisible(False)
        
        list_header.addWidget(self.file_count_label)
        list_header.addStretch()
        list_header.addWidget(self.clear_all_btn)
        
        # File list widget
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(200)
        self.file_list.setVisible(False)
        
        list_layout.addLayout(list_header)
        list_layout.addWidget(self.file_list)
        
        # Supported formats info
        self.formats_label = QLabel(f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}")
        self.formats_label.setStyleSheet("color: #666; font-size: 10px;")
        self.formats_label.setWordWrap(True)
        
        # Add all to main layout
        self.layout.addWidget(self.drop_zone)
        self.layout.addWidget(self.file_list_frame)
        self.layout.addWidget(self.formats_label)
        self.layout.addStretch()
        
    def setup_styling(self):
        """Setup widget styling"""
        # Drop zone styling
        self.drop_zone.setStyleSheet("""
            QFrame {
                border: 2px dashed #cccccc;
                border-radius: 10px;
                background-color: #f9f9f9;
                padding: 20px;
            }
            QFrame:hover {
                border-color: #4a90e2;
                background-color: #f0f8ff;
            }
        """)
        
        # Drop label styling
        self.drop_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                font-weight: normal;
            }
        """)
        
        # Browse button styling
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
        """)
        
        # Clear button styling
        self.clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        # File list styling
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
                selection-background-color: #e3f2fd;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            # Check if any files have supported extensions
            urls = event.mimeData().urls()
            has_supported = any(
                Path(url.toLocalFile()).suffix.lower() in self.SUPPORTED_EXTENSIONS
                for url in urls if url.isLocalFile()
            )
            
            if has_supported:
                event.acceptProposedAction()
                # Visual feedback
                self.drop_zone.setStyleSheet("""
                    QFrame {
                        border: 2px solid #4a90e2;
                        border-radius: 10px;
                        background-color: #e3f2fd;
                        padding: 20px;
                    }
                """)
            else:
                event.ignore()
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        """Handle drag leave events"""
        self.setup_styling()  # Reset styling
        
    def dropEvent(self, event: QDropEvent):
        """Handle drop events"""
        self.setup_styling()  # Reset styling
        
        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        files.append(str(file_path))
                        
            if files:
                self.add_files(files)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
            
    def browse_files(self):
        """Open file browser dialog"""
        file_filter = "Audio/Video Files ("
        file_filter += " ".join(f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS)
        file_filter += ");;All Files (*)"
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio/Video Files",
            "",
            file_filter
        )
        
        if files:
            self.add_files(files)
            
    def add_files(self, files: List[str]):
        """Add files to the list"""
        new_files = []
        for file_path in files:
            if file_path not in self.dropped_files:
                self.dropped_files.append(file_path)
                new_files.append(file_path)
                
        if new_files:
            self.update_file_list()
            self.files_dropped.emit(new_files)
            self.files_selected.emit(self.dropped_files.copy())
            
            # Call registered callbacks
            for callback in self.file_callbacks:
                try:
                    callback(new_files)
                except Exception as e:
                    logger.error(f"File callback error: {e}")
                    
    def remove_file(self, file_path: str):
        """Remove file from list"""
        if file_path in self.dropped_files:
            self.dropped_files.remove(file_path)
            self.update_file_list()
            self.file_removed.emit(file_path)
            
    def clear_all_files(self):
        """Clear all files from list"""
        self.dropped_files.clear()
        self.update_file_list()
        self.clear_all.emit()
        
    def update_file_list(self):
        """Update the file list display"""
        self.file_list.clear()
        
        if not self.dropped_files:
            self.file_list.setVisible(False)
            self.clear_all_btn.setVisible(False)
            self.file_count_label.setText("Selected Files (0)")
            return
            
        self.file_list.setVisible(True)
        self.clear_all_btn.setVisible(True)
        self.file_count_label.setText(f"Selected Files ({len(self.dropped_files)})")
        
        for file_path in self.dropped_files:
            item_widget = FileListItem(file_path, self.remove_file)
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())
            
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, item_widget)
            
    def get_files(self) -> List[str]:
        """Get list of selected files"""
        return self.dropped_files.copy()
        
    def add_file_callback(self, callback: Callable[[List[str]], None]):
        """Add callback for file selection"""
        self.file_callbacks.append(callback)
        
    def set_enabled(self, enabled: bool):
        """Enable/disable the widget"""
        self.setAcceptDrops(enabled)
        self.browse_btn.setEnabled(enabled)
        self.clear_all_btn.setEnabled(enabled)
        
        if enabled:
            self.drop_label.setText("🎵 Drag & Drop Audio/Video Files Here\n\nor click Browse to select files")
        else:
            self.drop_label.setText("🔒 Processing in progress...\nFile selection disabled")

class FileListItem(QWidget):
    """Individual file item in the list"""
    
    def __init__(self, file_path: str, remove_callback: Callable[[str], None]):
        super().__init__()
        self.file_path = file_path
        self.remove_callback = remove_callback
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the item UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # File info
        file_name = Path(self.file_path).name
        file_size = self.get_file_size()
        
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        name_label = QLabel(file_name)
        name_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        
        details_label = QLabel(f"{file_size} • {Path(self.file_path).suffix.upper()}")
        details_label.setStyleSheet("color: #666; font-size: 8px;")
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(details_label)
        
        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setMaximumSize(QSize(20, 20))
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
        """)
        remove_btn.clicked.connect(lambda: self.remove_callback(self.file_path))
        
        layout.addLayout(info_layout)
        layout.addStretch()
        layout.addWidget(remove_btn)
        
    def get_file_size(self) -> str:
        """Get formatted file size"""
        try:
            size_bytes = Path(self.file_path).stat().st_size
            
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes / 1024**2:.1f} MB"
            else:
                return f"{size_bytes / 1024**3:.1f} GB"
                
        except Exception:
            return "Unknown"

class TkinterDragDropWidget:
    """Tkinter version of drag & drop widget (fallback)"""
    
    def __init__(self, parent):
        if not HAS_TK_DND:
            raise ImportError("tkinterdnd2 required for tkinter drag & drop")
            
        self.parent = parent
        self.dropped_files = []
        self.file_callbacks = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup tkinter UI"""
        # Main frame
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Drop zone
        self.drop_frame = ttk.LabelFrame(self.main_frame, text="Drag & Drop Files", padding=20)
        self.drop_frame.pack(fill='x', pady=(0, 10))
        
        # Drop label
        self.drop_label = ttk.Label(
            self.drop_frame, 
            text="🎵 Drag & Drop Audio/Video Files Here\n\nor click Browse to select files",
            justify='center'
        )
        self.drop_label.pack(pady=10)
        
        # Browse button
        self.browse_btn = ttk.Button(
            self.drop_frame,
            text="📁 Browse Files",
            command=self.browse_files
        )
        self.browse_btn.pack()
        
        # File list
        self.file_listbox = tkinter.Listbox(self.main_frame, height=8)
        self.file_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        # Control buttons
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill='x')
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_all_files)
        self.clear_btn.pack(side='right')
        
        # Setup drag & drop
        self.drop_frame.drop_target_register(tkdnd.DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
        
    def on_drop(self, event):
        """Handle file drop"""
        files = []
        for file_path in event.data.split():
            file_path = file_path.strip('{}')  # Remove braces if present
            if Path(file_path).suffix.lower() in FileDropWidget.SUPPORTED_EXTENSIONS:
                files.append(file_path)
                
        if files:
            self.add_files(files)
            
    def browse_files(self):
        """Browse for files"""
        filetypes = [
            ("Audio/Video Files", " ".join(f"*{ext}" for ext in FileDropWidget.SUPPORTED_EXTENSIONS)),
            ("All Files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Audio/Video Files",
            filetypes=filetypes
        )
        
        if files:
            self.add_files(list(files))
            
    def add_files(self, files):
        """Add files to list"""
        new_files = []
        for file_path in files:
            if file_path not in self.dropped_files:
                self.dropped_files.append(file_path)
                new_files.append(file_path)
                
        if new_files:
            self.update_file_list()
            
            # Call callbacks
            for callback in self.file_callbacks:
                try:
                    callback(new_files)
                except Exception as e:
                    logger.error(f"File callback error: {e}")
                    
    def clear_all_files(self):
        """Clear all files"""
        self.dropped_files.clear()
        self.update_file_list()
        
    def update_file_list(self):
        """Update file list display"""
        self.file_listbox.delete(0, 'end')
        
        for file_path in self.dropped_files:
            file_name = Path(file_path).name
            self.file_listbox.insert('end', file_name)
            
    def get_files(self):
        """Get selected files"""
        return self.dropped_files.copy()
        
    def add_file_callback(self, callback):
        """Add file selection callback"""
        self.file_callbacks.append(callback)

def create_drag_drop_widget(parent=None):
    """Factory function to create appropriate drag & drop widget"""
    if HAS_QT:
        return FileDropWidget(parent)
    elif HAS_TK_DND:
        return TkinterDragDropWidget(parent)
    else:
        raise ImportError("No suitable drag & drop framework available")