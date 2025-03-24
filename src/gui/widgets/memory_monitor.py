"""
Memory monitoring widget for displaying GPU usage.
"""

import torch
import logging
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPalette

# Initialize logger
logger = logging.getLogger(__name__)

class MemoryMonitorWidget(QWidget):
    """Widget for monitoring and displaying GPU memory usage"""
    
    def __init__(self, parent=None, update_interval=5000):
        """
        Initialize the memory monitor widget
        
        Args:
            parent: Parent widget
            update_interval: Update interval in milliseconds
        """
        super().__init__(parent)
        
        # Store update interval
        self.update_interval = update_interval
        
        # Set up UI
        self.init_ui()
        
        # Set up update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(update_interval)
        
        # Initial update
        self.update_stats()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # Label for GPU info
        self.gpu_label = QLabel("GPU:")
        self.setLayout(layout)
        layout.addWidget(self.gpu_label)
        
        # Progress bar for memory usage
        self.memory_bar = QProgressBar()
        self.memory_bar.setMinimum(0)
        self.memory_bar.setMaximum(100)
        self.memory_bar.setValue(0)
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setFormat("%v%")
        self.memory_bar.setMinimumWidth(100)
        self.memory_bar.setMaximumWidth(150)
        layout.addWidget(self.memory_bar)
        
        # Label for memory details
        self.memory_label = QLabel("0.0 GB / 0.0 GB")
        layout.addWidget(self.memory_label)
        
        # Configure progress bar colors
        self.setup_progress_bar_style()
    
    def setup_progress_bar_style(self):
        """Set up progress bar style with color gradients based on usage"""
        # Style sheet for progress bar
        style = """
        QProgressBar {
            border: 1px solid #CCCCCC;
            border-radius: 3px;
            background-color: #F5F5F5;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #4CAF50,
                stop:0.6 #FFC107,
                stop:1 #F44336
            );
        }
        """
        
        self.memory_bar.setStyleSheet(style)
    
    def update_stats(self):
        """Update the memory statistics"""
        if not torch.cuda.is_available():
            self.gpu_label.setText("GPU: Not available")
            self.memory_bar.setValue(0)
            self.memory_bar.setFormat("N/A")
            self.memory_label.setText("")
            return
        
        try:
            # Get memory stats
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            
            # Get total GPU memory
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
            
            # If using multiple GPUs, show aggregated stats
            if torch.cuda.device_count() > 1:
                device_names = []
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    # Shorten the name if it's too long
                    if len(name) > 10:
                        name = name[:10] + "..."
                    device_names.append(name)
                
                gpu_info = "GPUs: " + ", ".join(device_names)
                self.gpu_label.setText(gpu_info)
                
                # Aggregate memory across all devices
                total_allocated = allocated
                total_reserved = reserved
                total_memory = total
                
                for i in range(torch.cuda.device_count()):
                    if i != device:  # Skip the current device that we already counted
                        allocated_i = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        reserved_i = torch.cuda.memory_reserved(i) / (1024 ** 3)
                        total_i = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                        
                        total_allocated += allocated_i
                        total_reserved += reserved_i
                        total_memory += total_i
                
                # Use the aggregated values
                allocated = total_allocated
                reserved = total_reserved
                total = total_memory
            else:
                # Single GPU
                device_name = torch.cuda.get_device_name(device)
                # Shorten the name if it's too long
                if len(device_name) > 20:
                    device_name = device_name[:20] + "..."
                self.gpu_label.setText(f"GPU: {device_name}")
            
            # Calculate percentages
            allocated_percent = min(int(allocated / total * 100), 100)
            reserved_percent = min(int(reserved / total * 100), 100)
            
            # Update UI
            self.memory_bar.setValue(allocated_percent)
            self.memory_label.setText(f"{allocated:.1f} GB / {total:.1f} GB")
            
            # Update progress bar color based on usage
            self.update_progress_bar_color(allocated_percent)
            
            # Update format based on usage
            self.memory_bar.setFormat(f"{allocated_percent}%")
            
            # Log extreme memory usage
            if allocated_percent > 90:
                logger.warning(f"High GPU memory usage: {allocated_percent}% ({allocated:.1f} GB / {total:.1f} GB)")
            
        except Exception as e:
            logger.error(f"Error updating memory stats: {str(e)}")
            self.memory_label.setText("Error")
    
    def update_progress_bar_color(self, percent):
        """Update the progress bar color based on the percentage"""
        # Already handled through the style sheet with gradient
        pass
    
    def set_update_interval(self, interval):
        """Set the update interval in milliseconds"""
        self.update_interval = interval
        self.timer.stop()
        self.timer.start(interval)
    
    def pause_updates(self):
        """Pause memory updates"""
        self.timer.stop()
    
    def resume_updates(self):
        """Resume memory updates"""
        self.timer.start(self.update_interval)