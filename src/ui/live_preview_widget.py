"""
Live Preview Widget for Real-time Transcription
Provides real-time transcription preview with visual feedback and editing capabilities.
"""

import asyncio
import threading
import logging
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
from queue import Queue, Empty
import time
from dataclasses import dataclass

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, 
        QPushButton, QProgressBar, QFrame, QSplitter, QScrollArea,
        QSlider, QSpinBox, QComboBox, QCheckBox, QGroupBox
    )
    from PyQt6.QtCore import (
        Qt, QTimer, pyqtSignal, QThread, QObject, QMutex, QMutexLocker
    )
    from PyQt6.QtGui import QFont, QTextCursor, QColor, QPalette
    HAS_QT = True
except ImportError:
    HAS_QT = False

try:
    import whisper
    import torch
    import numpy as np
    import soundfile as sf
except ImportError:
    whisper = None
    torch = None
    np = None
    sf = None

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionChunk:
    """Represents a chunk of transcribed text"""
    text: str
    start_time: float
    end_time: float
    confidence: float = 0.0
    is_final: bool = False
    chunk_id: int = 0

class LiveTranscriptionWorker(QObject):
    """Worker thread for live transcription processing"""
    
    chunk_ready = pyqtSignal(object)  # TranscriptionChunk
    progress_updated = pyqtSignal(float)  # Progress percentage
    error_occurred = pyqtSignal(str)  # Error message
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.is_running = False
        self.audio_queue = Queue()
        self.mutex = QMutex()
        
    def load_model(self, model_name: str = "base"):
        """Load Whisper model"""
        try:
            if whisper is None:
                raise ImportError("Whisper not available")
                
            self.model = whisper.load_model(model_name)
            logger.info(f"Loaded Whisper model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.error_occurred.emit(f"Failed to load model: {e}")
            return False
    
    def start_transcription(self):
        """Start transcription processing"""
        with QMutexLocker(self.mutex):
            self.is_running = True
            
        # Start processing loop
        threading.Thread(target=self._processing_loop, daemon=True).start()
    
    def stop_transcription(self):
        """Stop transcription processing"""
        with QMutexLocker(self.mutex):
            self.is_running = False
    
    def add_audio_chunk(self, audio_data: np.ndarray, chunk_id: int):
        """Add audio chunk for processing"""
        self.audio_queue.put((audio_data, chunk_id))
    
    def _processing_loop(self):
        """Main processing loop"""
        chunk_id = 0
        
        while True:
            with QMutexLocker(self.mutex):
                if not self.is_running:
                    break
            
            try:
                # Get audio chunk
                audio_data, chunk_id = self.audio_queue.get(timeout=1.0)
                
                # Transcribe chunk
                if self.model is not None:
                    result = self.model.transcribe(
                        audio_data,
                        fp16=torch.cuda.is_available(),
                        language=None,  # Auto-detect
                        task='transcribe',
                        verbose=False
                    )
                    
                    # Extract transcription
                    text = result.get('text', '').strip()
                    segments = result.get('segments', [])
                    
                    if text and segments:
                        # Create transcription chunk
                        first_segment = segments[0]
                        last_segment = segments[-1]
                        
                        chunk = TranscriptionChunk(
                            text=text,
                            start_time=first_segment.get('start', 0.0),
                            end_time=last_segment.get('end', 0.0),
                            confidence=np.mean([s.get('avg_logprob', -1.0) for s in segments]),
                            is_final=True,
                            chunk_id=chunk_id
                        )
                        
                        self.chunk_ready.emit(chunk)
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                self.error_occurred.emit(str(e))

class AudioVisualizationWidget(QWidget):
    """Widget for audio waveform visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.current_position = 0.0
        self.duration = 0.0
        self.setMinimumHeight(60)
        self.setMaximumHeight(100)
        
    def set_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """Set audio data for visualization"""
        self.audio_data = audio_data
        self.duration = len(audio_data) / sample_rate
        self.update()
        
    def set_position(self, position: float):
        """Set current playback position"""
        self.current_position = position
        self.update()
        
    def paintEvent(self, event):
        """Paint the waveform"""
        if not HAS_QT or self.audio_data is None:
            return
            
        from PyQt6.QtGui import QPainter, QPen
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        # Draw waveform
        width = self.width()
        height = self.height()
        mid_y = height // 2
        
        if len(self.audio_data) > 0:
            # Downsample for display
            samples_per_pixel = max(1, len(self.audio_data) // width)
            
            painter.setPen(QPen(QColor(70, 130, 180), 1))
            
            for x in range(width):
                start_sample = x * samples_per_pixel
                end_sample = min(start_sample + samples_per_pixel, len(self.audio_data))
                
                if start_sample < len(self.audio_data):
                    chunk = self.audio_data[start_sample:end_sample]
                    if len(chunk) > 0:
                        amplitude = np.max(np.abs(chunk)) if len(chunk) > 0 else 0
                        bar_height = int(amplitude * (height - 4) / 2)
                        
                        painter.drawLine(x, mid_y - bar_height, x, mid_y + bar_height)
        
        # Draw position indicator
        if self.duration > 0:
            pos_x = int((self.current_position / self.duration) * width)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawLine(pos_x, 0, pos_x, height)

class LivePreviewWidget(QWidget):
    """Main live preview widget"""
    
    # Signals
    transcription_updated = pyqtSignal(str)
    position_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        if not HAS_QT:
            raise ImportError("PyQt6 required for live preview")
            
        super().__init__(parent)
        
        self.audio_file_path = None
        self.audio_data = None
        self.sample_rate = 16000
        self.chunk_size = 30  # seconds
        self.transcription_chunks: List[TranscriptionChunk] = []
        
        # Worker thread
        self.worker_thread = QThread()
        self.worker = LiveTranscriptionWorker()
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.chunk_ready.connect(self.on_transcription_chunk)
        self.worker.progress_updated.connect(self.on_progress_update)
        self.worker.error_occurred.connect(self.on_error)
        
        self.worker_thread.start()
        
        self.setup_ui()
        self.setup_timers()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Controls section
        controls_group = QGroupBox("Live Preview Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(self.load_model)
        
        # Chunk size setting
        chunk_label = QLabel("Chunk Size (s):")
        self.chunk_spinbox = QSpinBox()
        self.chunk_spinbox.setRange(10, 60)
        self.chunk_spinbox.setValue(30)
        self.chunk_spinbox.valueChanged.connect(self.on_chunk_size_changed)
        
        # Auto-scroll option
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        
        # Load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(lambda: self.load_model())
        
        controls_layout.addWidget(QLabel("Model:"))
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(chunk_label)
        controls_layout.addWidget(self.chunk_spinbox)
        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addWidget(self.load_model_btn)
        controls_layout.addStretch()
        
        # Audio visualization
        self.audio_viz = AudioVisualizationWidget()
        
        # Position slider
        position_layout = QHBoxLayout()
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.valueChanged.connect(self.on_position_changed)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFont(QFont("monospace"))
        
        position_layout.addWidget(self.position_slider)
        position_layout.addWidget(self.time_label)
        
        # Transcription display
        transcription_group = QGroupBox("Live Transcription")
        transcription_layout = QVBoxLayout(transcription_group)
        
        # Text display with syntax highlighting
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setFont(QFont("Arial", 11))
        self.transcription_text.setMinimumHeight(200)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        
        transcription_layout.addWidget(self.transcription_text)
        transcription_layout.addWidget(self.progress_bar)
        transcription_layout.addWidget(self.status_label)
        
        # Add all sections to main layout
        layout.addWidget(controls_group)
        layout.addWidget(self.audio_viz)
        layout.addLayout(position_layout)
        layout.addWidget(transcription_group)
        
        # Initially disable until audio is loaded
        self.set_enabled(False)
        
    def setup_timers(self):
        """Setup update timers"""
        # Position update timer
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position)
        self.position_timer.setInterval(100)  # 10 FPS
        
        # Transcription processing timer
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_next_chunk)
        self.processing_timer.setInterval(1000)  # Every second
        
    def load_audio_file(self, file_path: str):
        """Load audio file for preview"""
        try:
            if sf is None:
                raise ImportError("soundfile not available")
                
            self.audio_file_path = Path(file_path)
            self.status_label.setText("Loading audio...")
            
            # Load audio data
            self.audio_data, self.sample_rate = sf.read(str(file_path))
            
            # Convert to mono if stereo
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
                
            # Normalize audio
            self.audio_data = self.audio_data.astype(np.float32)
            max_val = np.max(np.abs(self.audio_data))
            if max_val > 0:
                self.audio_data /= max_val
            
            duration = len(self.audio_data) / self.sample_rate
            
            # Setup UI for loaded audio
            self.position_slider.setMaximum(int(duration * 1000))  # milliseconds
            self.audio_viz.set_audio_data(self.audio_data, self.sample_rate)
            self.time_label.setText(f"00:00 / {self.format_time(duration)}")
            
            self.set_enabled(True)
            self.status_label.setText(f"Audio loaded: {duration:.1f}s, {self.sample_rate}Hz")
            
            # Clear previous transcription
            self.transcription_chunks.clear()
            self.transcription_text.clear()
            
            logger.info(f"Loaded audio: {file_path} ({duration:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            self.status_label.setText(f"Error: {e}")
            return False
    
    def load_model(self):
        """Load transcription model"""
        model_name = self.model_combo.currentText()
        self.status_label.setText(f"Loading model: {model_name}...")
        self.load_model_btn.setEnabled(False)
        
        # Load in worker thread
        success = self.worker.load_model(model_name)
        
        if success:
            self.status_label.setText(f"Model loaded: {model_name}")
        
        self.load_model_btn.setEnabled(True)
    
    def start_live_preview(self):
        """Start live transcription preview"""
        if self.audio_data is None:
            return
            
        self.worker.start_transcription()
        self.position_timer.start()
        self.processing_timer.start()
        self.progress_bar.setVisible(True)
        self.status_label.setText("Processing...")
        
        # Start feeding audio chunks
        self._feed_audio_chunks()
        
    def stop_live_preview(self):
        """Stop live transcription preview"""
        self.worker.stop_transcription()
        self.position_timer.stop()
        self.processing_timer.stop()
        self.progress_bar.setVisible(False)
        self.status_label.setText("Stopped")
        
    def _feed_audio_chunks(self):
        """Feed audio chunks to transcription worker"""
        if self.audio_data is None:
            return
            
        chunk_samples = int(self.chunk_size * self.sample_rate)
        total_chunks = (len(self.audio_data) + chunk_samples - 1) // chunk_samples
        
        for chunk_id in range(total_chunks):
            start_idx = chunk_id * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(self.audio_data))
            
            chunk_data = self.audio_data[start_idx:end_idx]
            self.worker.add_audio_chunk(chunk_data, chunk_id)
            
    def on_transcription_chunk(self, chunk: TranscriptionChunk):
        """Handle new transcription chunk"""
        self.transcription_chunks.append(chunk)
        self.update_transcription_display()
        
    def on_progress_update(self, progress: float):
        """Handle progress update"""
        self.progress_bar.setValue(int(progress * 100))
        
    def on_error(self, error: str):
        """Handle transcription error"""
        self.status_label.setText(f"Error: {error}")
        logger.error(f"Transcription error: {error}")
        
    def on_position_changed(self, value: int):
        """Handle position slider change"""
        position = value / 1000.0  # Convert from milliseconds
        self.audio_viz.set_position(position)
        
        if self.audio_data is not None:
            duration = len(self.audio_data) / self.sample_rate
            self.time_label.setText(f"{self.format_time(position)} / {self.format_time(duration)}")
            
        self.position_changed.emit(position)
        
    def on_chunk_size_changed(self, value: int):
        """Handle chunk size change"""
        self.chunk_size = value
        
    def update_position(self):
        """Update position display"""
        # This would be updated by an audio player
        pass
        
    def process_next_chunk(self):
        """Process next audio chunk"""
        # This would trigger the next chunk processing
        pass
        
    def update_transcription_display(self):
        """Update transcription text display"""
        # Sort chunks by start time
        sorted_chunks = sorted(self.transcription_chunks, key=lambda c: c.start_time)
        
        # Build combined text with timing and confidence info
        combined_text = ""
        for chunk in sorted_chunks:
            confidence_indicator = "✓" if chunk.confidence > -0.5 else "?"
            timing = f"[{self.format_time(chunk.start_time)}]"
            
            combined_text += f"{timing} {confidence_indicator} {chunk.text}\n\n"
            
        self.transcription_text.setPlainText(combined_text)
        
        # Auto-scroll to bottom if enabled
        if self.auto_scroll_checkbox.isChecked():
            cursor = self.transcription_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.transcription_text.setTextCursor(cursor)
            
        # Emit signal
        self.transcription_updated.emit(combined_text)
        
    def format_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def set_enabled(self, enabled: bool):
        """Enable/disable controls"""
        self.position_slider.setEnabled(enabled)
        
    def get_transcription_text(self) -> str:
        """Get current transcription text"""
        return self.transcription_text.toPlainText()
        
    def export_transcription(self, file_path: str):
        """Export transcription to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.get_transcription_text())
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
            
    def closeEvent(self, event):
        """Handle widget close"""
        self.stop_live_preview()
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

# Factory function
def create_live_preview_widget(parent=None):
    """Create live preview widget"""
    if not HAS_QT:
        raise ImportError("PyQt6 required for live preview")
    return LivePreviewWidget(parent)