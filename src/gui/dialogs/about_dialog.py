"""
About dialog for displaying application information.
"""

import platform
import torch
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QTabWidget, QTextBrowser, QPushButton, 
                            QDialogButtonBox, QWidget, QFormLayout)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt

from src.version import VERSION, VERSION_DISPLAY

class AboutDialog(QDialog):
    """About dialog with application information across multiple tabs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("About Advanced Code Generator")
        self.setMinimumSize(600, 400)
        
        # You can use VERSION here
        print(VERSION)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Header with application info
        header_layout = QHBoxLayout()
        
        # App logo (if available)
        try:
            logo_label = QLabel()
            pixmap = QPixmap("assets/icons/app_icon.png")
            if not pixmap.isNull():
                logo_label.setPixmap(pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                header_layout.addWidget(logo_label)
        except Exception:
            pass  # No logo available
        
        # App title and version
        title_layout = QVBoxLayout()
        
        app_name_label = QLabel("Advanced Code Generator")
        app_name_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_layout.addWidget(app_name_label)
        
        version_label = QLabel(f"Version {VERSION_DISPLAY}")
        version_label.setFont(QFont("Arial", 10))
        title_layout.addWidget(version_label)
        
        # Description
        description_label = QLabel(
            "An intelligent code generation application using state-of-the-art language models."
        )
        description_label.setWordWrap(True)
        title_layout.addWidget(description_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Tab widget for additional information
        tab_widget = QTabWidget()
        
        # About tab
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        about_text = QTextBrowser()
        about_text.setOpenExternalLinks(True)
        about_text.setHtml(self.get_about_html())
        about_layout.addWidget(about_text)
        
        tab_widget.addTab(about_tab, "About")
        
        # System Info tab
        system_tab = QWidget()
        system_layout = QVBoxLayout(system_tab)
        
        system_text = QTextBrowser()
        system_text.setHtml(self.get_system_info_html())
        system_layout.addWidget(system_text)
        
        tab_widget.addTab(system_tab, "System Info")
        
        # License tab
        license_tab = QWidget()
        license_layout = QVBoxLayout(license_tab)
        
        license_text = QTextBrowser()
        license_text.setPlainText(self.get_license_text())
        license_layout.addWidget(license_text)
        
        tab_widget.addTab(license_tab, "License")
        
        # Credits tab
        credits_tab = QWidget()
        credits_layout = QVBoxLayout(credits_tab)
        
        credits_text = QTextBrowser()
        credits_text.setOpenExternalLinks(True)
        credits_text.setHtml(self.get_credits_html())
        credits_layout.addWidget(credits_text)
        
        tab_widget.addTab(credits_tab, "Credits")
        
        layout.addWidget(tab_widget)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_about_html(self):
        """Get HTML content for About tab"""
        return f"""
        <h2>Advanced Code Generator</h2>
        <p>Version {VERSION_DISPLAY}</p>
        
        <p>A powerful desktop application for generating high-quality code using large language models.</p>
        
        <h3>Features:</h3>
        <ul>
            <li>Advanced code editor with syntax highlighting</li>
            <li>Real-time streaming code generation</li>
            <li>Support for multiple programming languages</li>
            <li>Template-based code generation</li>
            <li>History tracking and management</li>
            <li>Model quantization and memory optimization</li>
        </ul>
        
        <p>Built with PyQt5 and Hugging Face Transformers</p>
        
        <p><a href="https://github.com/yourusername/advanced-code-generator">GitHub Repository</a></p>
        """
    
    def get_system_info_html(self):
        """Get HTML content for System Info tab"""
        # Get CUDA info if available
        cuda_info = "Not available"
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            
            cuda_info = f"CUDA {cuda_version}, {device_count} device(s) available<br>"
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
                cuda_info += f"Device {i}: {device_name}, {total_memory:.2f} GB<br>"
        
        # Get PyTorch info
        pytorch_version = torch.__version__
        pytorch_info = f"PyTorch {pytorch_version}"
        
        try:
            import transformers
            transformers_version = transformers.__version__
            transformers_info = f"Transformers {transformers_version}"
        except ImportError:
            transformers_info = "Not installed"
        
        try:
            from PyQt5.QtCore import QT_VERSION_STR
            qt_version = QT_VERSION_STR
        except ImportError:
            qt_version = "Unknown"
        
        return f"""
        <h2>System Information</h2>
        
        <h3>Software:</h3>
        <ul>
            <li><b>Operating System:</b> {platform.system()} {platform.version()}</li>
            <li><b>Python:</b> {platform.python_version()}</li>
            <li><b>PyQt:</b> {qt_version}</li>
            <li><b>PyTorch:</b> {pytorch_info}</li>
            <li><b>Transformers:</b> {transformers_info}</li>
        </ul>
        
        <h3>Hardware:</h3>
        <ul>
            <li><b>Processor:</b> {platform.processor()}</li>
            <li><b>Architecture:</b> {platform.machine()}</li>
            <li><b>CUDA:</b> {cuda_info}</li>
        </ul>
        """
    
    def get_license_text(self):
        """Get license text"""
        # Use a simple MIT license text as an example
        return """MIT License

Copyright (c) 2023 Advanced Code Generator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
    
    def get_credits_html(self):
        """Get HTML content for Credits tab"""
        return """
        <h2>Credits and Acknowledgments</h2>
        
        <h3>Libraries and Technologies:</h3>
        <ul>
            <li><a href="https://www.qt.io/">Qt</a> and <a href="https://riverbankcomputing.com/software/pyqt/">PyQt</a> - GUI framework</li>
            <li><a href="https://pytorch.org/">PyTorch</a> - Machine learning framework</li>
            <li><a href="https://huggingface.co/docs/transformers/">Hugging Face Transformers</a> - LLM interface library</li>
            <li><a href="https://www.riverbankcomputing.com/software/qscintilla/">QScintilla</a> - Advanced code editing component</li>
            <li><a href="https://github.com/psf/black">Black</a> and <a href="https://github.com/beautify-web/js-beautify">js-beautify</a> - Code formatting</li>
        </ul>
        
        <h3>LLM Models:</h3>
        <p>This application is designed to work with various LLM models available through the Hugging Face Hub. Each model has its own license and attribution requirements. Please refer to the model card for any model you use.</p>
        
        <h3>Development Team:</h3>
        <ul>
            <li>Project Lead: Advanced Code Generator Team</li>
            <li>Development: Advanced Code Generator Contributors</li>
            <li>Documentation: Advanced Code Generator Community</li>
        </ul>
        
        <h3>Special Thanks:</h3>
        <p>Special thanks to the open source community for their contributions to the libraries and technologies that make this application possible.</p>
        """