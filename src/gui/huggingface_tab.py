"""
Hugging Face model browser tab for the Advanced Code Generator.
Provides functionality to discover, filter and load models from Hugging Face.
"""

import os
import sys
import logging
import json
import time
import torch
import threading
import ctypes
from typing import Dict, List, Any, Optional, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QComboBox, QPushButton, QGroupBox, QFormLayout, 
                           QListWidget, QListWidgetItem, QSplitter, QTextEdit,
                           QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                           QMenu, QAction, QMessageBox, QFileDialog, QCheckBox,
                           QProgressBar, QScrollArea, QFrame, QToolTip, QSpinBox,
                           QRadioButton, QButtonGroup, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QUrl, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap, QDesktopServices
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

# Logger for this module
logger = logging.getLogger(__name__)

# Constants
HUGGINGFACE_API_URL = "https://huggingface.co/api"
MODELS_PER_PAGE = 20
MODEL_CATEGORIES = [
    "All Categories",
    "Text Generation",
    "Code Generation",
    "Conversational",
    "Embeddings",
    "Translation",
    "Summarization",
]
MODEL_SIZES = [
    "All Sizes",
    "< 500MB",
    "500MB - 2GB",
    "2GB - 7GB",
    "7GB - 15GB",
    "> 15GB",
]
MODEL_FRAMEWORKS = [
    "All Frameworks",
    "PyTorch",
    "TensorFlow",
    "Flax",
    "ONNX",
]
MODEL_LICENSES = [
    "All Licenses",
    "Open Source",
    "Commercial Use OK",
    "Research Only",
]


class ModelSearchThread(QThread):
    """Thread for searching models in the Hugging Face Hub"""
    
    # Signals
    results_ready = pyqtSignal(list, int)  # results, total_count
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)  # 0-100
    
    def __init__(self, search_params):
        super().__init__()
        self.search_params = search_params
        self.is_cancelled = False
        
    def run(self):
        """Run the search thread"""
        try:
            import requests
            
            # Calculate system specs to filter models by compatibility
            self.progress_updated.emit(10)
            system_specs = self.get_system_specs()
            
            # Prepare search parameters
            params = {
                "search": self.search_params.get("query", ""),
                "limit": MODELS_PER_PAGE,
                "offset": self.search_params.get("page", 0) * MODELS_PER_PAGE,
                "filter": "text-generation",  # Default filter for code generation
                "sort": "downloads",
            }
            
            # Add category filter
            category = self.search_params.get("category")
            if category and category != "All Categories":
                if category == "Code Generation":
                    params["filter"] = "text-generation"
                    params["search"] = (params["search"] + " code").strip()
                else:
                    params["filter"] = category.lower().replace(" ", "-")
            
            # Add framework filter
            framework = self.search_params.get("framework")
            if framework and framework != "All Frameworks":
                params["library"] = framework.lower()
            
            # Add license filter (will be applied client-side after getting results)
            
            self.progress_updated.emit(30)
            
            # Make API request
            url = f"{HUGGINGFACE_API_URL}/models"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            self.progress_updated.emit(60)
            
            # Process results
            results = response.json()
            
            # Apply size and hardware compatibility filters
            filtered_results = []
            size_filter = self.search_params.get("size")
            license_filter = self.search_params.get("license")
            
            for model in results:
                # Skip if cancelled
                if self.is_cancelled:
                    return
                
                # Apply size filter
                if size_filter and size_filter != "All Sizes":
                    model_size_mb = self.estimate_model_size(model)
                    
                    if size_filter == "< 500MB" and model_size_mb >= 500:
                        continue
                    elif size_filter == "500MB - 2GB" and (model_size_mb < 500 or model_size_mb >= 2000):
                        continue
                    elif size_filter == "2GB - 7GB" and (model_size_mb < 2000 or model_size_mb >= 7000):
                        continue
                    elif size_filter == "7GB - 15GB" and (model_size_mb < 7000 or model_size_mb >= 15000):
                        continue
                    elif size_filter == "> 15GB" and model_size_mb < 15000:
                        continue
                
                # Apply license filter
                if license_filter and license_filter != "All Licenses":
                    model_license = model.get("license", "").lower()
                    
                    if license_filter == "Open Source" and not any(lic in model_license for lic in ["mit", "apache", "gpl", "bsd", "cc", "mozilla", "lgpl"]):
                        continue
                    elif license_filter == "Commercial Use OK" and any(lic in model_license for lic in ["non-commercial", "noncommercial", "research", "no commercial"]):
                        continue
                    elif license_filter == "Research Only" and not any(lic in model_license for lic in ["research", "non-commercial", "noncommercial", "academic"]):
                        continue
                
                # Check system compatibility
                compatibility_score = self.check_model_compatibility(model, system_specs)
                model["compatibility_score"] = compatibility_score
                
                # Add usage type
                model["best_usage"] = self.determine_best_usage(model)
                
                filtered_results.append(model)
            
            self.progress_updated.emit(90)
            
            # Sort by compatibility and then downloads
            filtered_results.sort(key=lambda x: (-x.get("compatibility_score", 0), -x.get("downloads", 0)))
            
            # Emit results
            self.results_ready.emit(filtered_results, len(filtered_results))
            
            self.progress_updated.emit(100)
            
        except Exception as e:
            logger.error(f"Error searching models: {str(e)}")
            self.error_occurred.emit(f"Error searching models: {str(e)}")
    
    def cancel(self):
        """Cancel the search"""
        self.is_cancelled = True
    
    def get_system_specs(self):
        """Get system specifications for compatibility checking"""
        specs = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_memory_gb": 0,
            "cpu_count": os.cpu_count() or 4,
            "system_memory_gb": self.get_system_memory_gb(),
        }
        
        # Get GPU memory if available
        if specs["cuda_available"]:
            try:
                device = torch.cuda.current_device()
                specs["gpu_memory_gb"] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            except Exception as e:
                logger.warning(f"Error getting GPU memory: {str(e)}")
        
        return specs
    
    def get_system_memory_gb(self):
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback if psutil not available
            import platform
            if platform.system() == "Windows":
                # Windows fallback
                from ctypes import Structure, c_uint64, c_long, POINTER, byref, windll
                class MEMORYSTATUSEX(Structure):
                    _fields_ = [
                        ("dwLength", c_long),
                        ("dwMemoryLoad", c_long),
                        ("ullTotalPhys", c_uint64),
                        ("ullAvailPhys", c_uint64),
                        ("ullTotalPageFile", c_uint64),
                        ("ullAvailPageFile", c_uint64),
                        ("ullTotalVirtual", c_uint64),
                        ("ullAvailVirtual", c_uint64),
                        ("ullAvailExtendedVirtual", c_uint64),
                    ]
                
                memoryStatus = MEMORYSTATUSEX()
                memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                windll.kernel32.GlobalMemoryStatusEx(byref(memoryStatus))
                return memoryStatus.ullTotalPhys / (1024**3)
            else:
                # Linux/macOS fallback
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemTotal' in line:
                                return int(line.split()[1]) / (1024**2)
                except:
                    # Return a default value if all fails
                    return 8.0
    
    def estimate_model_size(self, model):
        """Estimate model size in MB based on metadata"""
        # Try to get sibling size information
        try:
            if "siblings" in model:
                total_size = 0
                for sibling in model["siblings"]:
                    if "size" in sibling:
                        total_size += sibling["size"]
                
                if total_size > 0:
                    return total_size / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            pass
        
        # Fallback: estimate from model metadata
        try:
            model_type = model.get("model_type", "").lower()
            tags = model.get("tags", [])
            
            # Check for size related tags
            for tag in tags:
                tag = tag.lower()
                if "7b" in tag:
                    return 14000  # ~14GB
                elif "13b" in tag:
                    return 26000  # ~26GB
                elif "70b" in tag:
                    return 140000  # ~140GB
                elif "1.5b" in tag or "1b" in tag:
                    return 3000  # ~3GB
                elif "125m" in tag:
                    return 250  # ~250MB

            # Base size on architecture
            if "gpt-j" in model_type:
                return 6000  # ~6GB
            elif "llama" in model_type:
                return 13000  # ~13GB
            elif "t5" in model_type:
                return 2000  # ~2GB
            elif "bart" in model_type:
                return 1500  # ~1.5GB
            elif "bert" in model_type:
                return 500  # ~500MB
            elif "gpt2" in model_type:
                return 1500  # ~1.5GB
            elif "gpt-neo" in model_type:
                return 6000  # ~6GB
            elif "deberta" in model_type:
                return 600  # ~600MB
        except Exception:
            pass
        
        # Default size estimate based on downloads (popular models tend to be larger)
        downloads = model.get("downloads", 0)
        if downloads > 1000000:
            return 5000  # ~5GB
        elif downloads > 500000:
            return 3000  # ~3GB
        elif downloads > 100000:
            return 1500  # ~1.5GB
        else:
            return 800  # ~800MB
    
    def check_model_compatibility(self, model, system_specs):
        """Check model compatibility with the system and return a score (0-100)"""
        score = 100
        model_size_mb = self.estimate_model_size(model)
        model_size_gb = model_size_mb / 1024
        
        # Check if model fits in GPU memory (with 20% overhead for processing)
        if system_specs["cuda_available"]:
            if model_size_gb * 1.2 > system_specs["gpu_memory_gb"]:
                # Model doesn't fit in GPU memory, reduce score
                score -= 30
                
                # Check for quantized versions
                tags = [tag.lower() for tag in model.get("tags", [])]
                if any(q in " ".join(tags) for q in ["quantized", "ggml", "gguf", "gptq", "8bit", "4bit"]):
                    # Quantized versions available, improve score
                    score += 15
        else:
            # No GPU available, large models will be slow
            if model_size_gb > 2:
                score -= 40
            elif model_size_gb > 1:
                score -= 20
        
        # Check if model requires more than 70% of system memory
        if model_size_gb > system_specs["system_memory_gb"] * 0.7:
            score -= 30
        
        # Check for high memory usage
        if model_size_gb > 12:
            score -= 10  # Very large model
        
        # Check for low-end hardware optimizations
        tags = [tag.lower() for tag in model.get("tags", [])]
        if any(opt in " ".join(tags) for opt in ["optimized", "efficient", "tiny", "small", "compact"]):
            score += 10
        
        # Adjust score based on popularity (popular models tend to be more robust)
        downloads = model.get("downloads", 0)
        if downloads > 500000:
            score += 10
        elif downloads > 100000:
            score += 5
        
        # Ensure score is within range
        return max(0, min(100, score))
    
    def determine_best_usage(self, model):
        """Determine the best usage for the model"""
        # Get model metadata
        model_id = model.get("id", "").lower()
        model_name = model.get("modelId", "").lower()
        tags = [tag.lower() for tag in model.get("tags", [])]
        pipeline_tag = model.get("pipeline_tag", "").lower()
        
        tag_str = " ".join(tags)
        
        # Check for code generation indicators
        if any(code_kw in tag_str or code_kw in model_id or code_kw in model_name 
               for code_kw in ["code", "starcoder", "codegen", "codellama", "coder"]):
            return "Code Generation"
        
        # Check for chat/conversation indicators
        if any(chat_kw in tag_str or chat_kw in model_id or chat_kw in model_name 
               for chat_kw in ["chat", "conversation", "instruct", "assistant", "dialogue"]):
            return "Conversation & Chat"
        
        # Check for specific task indicators
        if "text-to-image" in tag_str or pipeline_tag == "text-to-image":
            return "Text-to-Image Generation"
        elif "summarization" in tag_str or pipeline_tag == "summarization":
            return "Text Summarization"
        elif "translation" in tag_str or pipeline_tag == "translation":
            return "Translation"
        elif "question-answering" in tag_str or pipeline_tag == "question-answering":
            return "Question Answering"
        
        # Default to text generation
        return "Text Generation"


class HuggingFaceModelWidget(QWidget):
    """Widget to display a Hugging Face model with details"""
    
    model_selected = pyqtSignal(dict)
    
    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Main container with border
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setFrameShadow(QFrame.Raised)
        container.setLineWidth(1)
        container.setStyleSheet("background-color: #F8F9FA; border-radius: 8px;")
        
        container_layout = QVBoxLayout()
        container_layout.setSpacing(8)
        
        # Model name and author
        name_layout = QHBoxLayout()
        model_name = QLabel(f"<b>{self.model_data.get('modelId', '')}</b>")
        model_name.setStyleSheet("font-size: 16px;")
        name_layout.addWidget(model_name)
        name_layout.addStretch()
        
        author_label = QLabel(f"by {self.model_data.get('author', '')}")
        author_label.setStyleSheet("color: #6C757D;")
        name_layout.addWidget(author_label)
        
        container_layout.addLayout(name_layout)
        
        # Model details
        details_layout = QHBoxLayout()
        
        # Left side: Usage, compatibility, downloads
        left_details = QVBoxLayout()
        
        # Best usage
        usage_label = QLabel(f"<b>Best for:</b> {self.model_data.get('best_usage', 'Text Generation')}")
        left_details.addWidget(usage_label)
        
        # Compatibility score
        compat_score = self.model_data.get("compatibility_score", 0)
        compat_color = "#28A745" if compat_score >= 70 else "#FFC107" if compat_score >= 40 else "#DC3545"
        
        compat_layout = QHBoxLayout()
        compat_layout.setSpacing(5)
        compat_label = QLabel("<b>Compatibility:</b>")
        compat_layout.addWidget(compat_label)
        
        compat_bar = QProgressBar()
        compat_bar.setRange(0, 100)
        compat_bar.setValue(compat_score)
        compat_bar.setTextVisible(True)
        compat_bar.setFixedHeight(16)
        compat_bar.setFixedWidth(100)
        compat_bar.setStyleSheet(f"QProgressBar {{ border: 1px solid #CCC; border-radius: 4px; text-align: center; }} "
                               f"QProgressBar::chunk {{ background-color: {compat_color}; }}")
        compat_layout.addWidget(compat_bar)
        
        left_details.addLayout(compat_layout)
        
        # Downloads
        downloads = self.model_data.get("downloads", 0)
        downloads_str = f"{downloads:,}" if downloads < 1000000 else f"{downloads/1000000:.1f}M"
        downloads_label = QLabel(f"<b>Downloads:</b> {downloads_str}")
        left_details.addWidget(downloads_label)
        
        details_layout.addLayout(left_details)
        details_layout.addSpacing(20)
        
        # Right side: Size, frameworks, license
        right_details = QVBoxLayout()
        
        # Estimated size
        model_size_mb = self.estimate_model_size(self.model_data)
        size_str = f"{model_size_mb/1024:.1f} GB" if model_size_mb >= 1024 else f"{model_size_mb:.0f} MB"
        size_label = QLabel(f"<b>Size:</b> {size_str}")
        right_details.addWidget(size_label)
        
        # Framework
        library = self.model_data.get("library", {}).get("name", "Not specified")
        framework_label = QLabel(f"<b>Framework:</b> {library.capitalize()}")
        right_details.addWidget(framework_label)
        
        # License
        license_text = self.model_data.get("license", "Not specified")
        license_label = QLabel(f"<b>License:</b> {license_text}")
        right_details.addWidget(license_label)
        
        details_layout.addLayout(right_details)
        container_layout.addLayout(details_layout)
        
        # Description
        desc = self.model_data.get("description", "No description available.")
        if len(desc) > 200:
            desc = desc[:197] + "..."
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #495057;")
        container_layout.addWidget(desc_label)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        # Open in browser button
        browser_btn = QPushButton("View on HF")
        browser_btn.setToolTip("Open model page on Hugging Face")
        browser_btn.clicked.connect(self.open_in_browser)
        actions_layout.addWidget(browser_btn)
        
        # Load model button
        load_btn = QPushButton("Load Model")
        load_btn.setToolTip("Load this model in the application")
        load_btn.clicked.connect(self.select_model)
        load_btn.setStyleSheet("background-color: #007BFF; color: white;")
        actions_layout.addWidget(load_btn)
        
        container_layout.addLayout(actions_layout)
        container.setLayout(container_layout)
        
        layout.addWidget(container)
        self.setLayout(layout)
    
    def estimate_model_size(self, model):
        """Estimate model size in MB (simplified version of the thread method)"""
        # Try to get sibling size information
        try:
            if "siblings" in model:
                total_size = 0
                for sibling in model["siblings"]:
                    if "size" in sibling:
                        total_size += sibling["size"]
                
                if total_size > 0:
                    return total_size / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            pass
        
        # Fallback: estimate from model metadata
        try:
            model_type = model.get("model_type", "").lower()
            tags = model.get("tags", [])
            tag_str = " ".join(tags).lower()
            
            # Check for size related tags
            if "7b" in tag_str:
                return 14000  # ~14GB
            elif "13b" in tag_str:
                return 26000  # ~26GB
            elif "70b" in tag_str:
                return 140000  # ~140GB
            elif "1.5b" in tag_str or "1b" in tag_str:
                return 3000  # ~3GB
            elif "125m" in tag_str:
                return 250  # ~250MB
            
            # Check for quantized models
            if any(q in tag_str for q in ["4bit", "4-bit"]):
                # 4-bit quantization ~= 25% of full size
                return self.estimate_base_model_size(model) * 0.25
            elif any(q in tag_str for q in ["8bit", "8-bit", "int8"]):
                # 8-bit quantization ~= 50% of full size
                return self.estimate_base_model_size(model) * 0.5
            elif "gptq" in tag_str:
                return self.estimate_base_model_size(model) * 0.3
            elif any(q in tag_str for q in ["ggml", "gguf"]):
                return self.estimate_base_model_size(model) * 0.4
        except Exception:
            pass
        
        return self.estimate_base_model_size(model)
    
    def estimate_base_model_size(self, model):
        """Estimate the base model size without quantization"""
        model_id = model.get("modelId", "").lower()
        
        # Base size on architecture name
        if "gpt-j" in model_id:
            return 6000  # ~6GB
        elif "llama" in model_id:
            return 13000  # ~13GB
        elif "t5" in model_id:
            return 2000  # ~2GB
        elif "bart" in model_id:
            return 1500  # ~1.5GB
        elif "bert" in model_id:
            return 500  # ~500MB
        elif "gpt2" in model_id:
            return 1500  # ~1.5GB
        elif "gpt-neo" in model_id:
            return 6000  # ~6GB
        elif "deberta" in model_id:
            return 600  # ~600MB
        
        # Default size estimate
        downloads = model.get("downloads", 0)
        if downloads > 1000000:
            return 5000  # ~5GB
        elif downloads > 500000:
            return 3000  # ~3GB
        elif downloads > 100000:
            return 1500  # ~1.5GB
        else:
            return 800  # ~800MB
    
    def open_in_browser(self):
        """Open the model page in a web browser"""
        model_id = self.model_data.get("id", "")
        url = f"https://huggingface.co/{model_id}"
        QDesktopServices.openUrl(QUrl(url))
    
    def select_model(self):
        """Emit signal to load this model"""
        self.model_selected.emit(self.model_data)


class HuggingFaceTab(QWidget):
    """Tab for browsing Hugging Face models"""
    
    # Signals
    model_selected = pyqtSignal(str)  # model_id
    
    def __init__(self, model_manager=None, parent=None):
        super().__init__(parent)
        
        # Store model manager reference
        self.model_manager = model_manager
        
        # State variables
        self.current_page = 0
        self.total_results = 0
        self.is_searching = False
        self.search_thread = None
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Search and filters section
        search_group = QGroupBox("Find Models")
        search_layout = QVBoxLayout()
        
        # Search query and button
        search_bar_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search Hugging Face models...")
        self.search_input.returnPressed.connect(self.search_models)
        search_bar_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_models)
        search_bar_layout.addWidget(self.search_button)
        
        search_layout.addLayout(search_bar_layout)
        
        # Filters
        filters_layout = QHBoxLayout()
        
        # Category filter
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(MODEL_CATEGORIES)
        cat_layout.addWidget(self.category_combo)
        filters_layout.addLayout(cat_layout)
        
        # Size filter
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(MODEL_SIZES)
        size_layout.addWidget(self.size_combo)
        filters_layout.addLayout(size_layout)
        
        # Framework filter
        framework_layout = QHBoxLayout()
        framework_layout.addWidget(QLabel("Framework:"))
        self.framework_combo = QComboBox()
        self.framework_combo.addItems(MODEL_FRAMEWORKS)
        framework_layout.addWidget(self.framework_combo)
        filters_layout.addLayout(framework_layout)
        
        # License filter
        license_layout = QHBoxLayout()
        license_layout.addWidget(QLabel("License:"))
        self.license_combo = QComboBox()
        self.license_combo.addItems(MODEL_LICENSES)
        license_layout.addWidget(self.license_combo)
        filters_layout.addLayout(license_layout)
        
        search_layout.addLayout(filters_layout)
        
        # Add a recommendation button at the top of the search group
        recommendation_layout = QHBoxLayout()
        recommendation_layout.addStretch()

        self.recommendation_button = QPushButton("Get Model Recommendations")
        self.recommendation_button.setStyleSheet(
            "background-color: #28a745; color: white; padding: 8px 16px; font-weight: bold;"
        )
        self.recommendation_button.setToolTip("Get recommendations based on your needs and system capabilities")
        self.recommendation_button.clicked.connect(self.show_recommendation_dialog)
        recommendation_layout.addWidget(self.recommendation_button)

        search_layout.addLayout(recommendation_layout)
        
        # System compatibility options
        compat_layout = QHBoxLayout()
        
        self.compat_check = QCheckBox("Show only models compatible with my system")
        self.compat_check.setChecked(True)
        compat_layout.addWidget(self.compat_check)
        
        # Display options
        self.show_quantized = QCheckBox("Prioritize quantized versions")
        self.show_quantized.setToolTip("Show 4-bit/8-bit quantized versions when available")
        self.show_quantized.setChecked(True)
        compat_layout.addWidget(self.show_quantized)
        
        search_layout.addLayout(compat_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        search_layout.addWidget(self.progress_bar)
        
        search_group.setLayout(search_layout)
        main_layout.addWidget(search_group)
        
        # Results area
        results_group = QGroupBox("Model Results")
        results_layout = QVBoxLayout()
        
        # Scroll area for results
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results_scroll.setFrameShape(QFrame.NoFrame)
        
        # Container for result widgets
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(10)
        self.results_container.setLayout(self.results_layout)
        self.results_scroll.setWidget(self.results_container)
        
        results_layout.addWidget(self.results_scroll)
        
        # Pagination and info
        pagination_layout = QHBoxLayout()
        
        # Results count
        self.results_count_label = QLabel("No results found")
        pagination_layout.addWidget(self.results_count_label)
        
        pagination_layout.addStretch()
        
        # Pagination controls
        self.prev_button = QPushButton("< Previous")
        self.prev_button.clicked.connect(self.prev_page)
        self.prev_button.setEnabled(False)
        pagination_layout.addWidget(self.prev_button)
        
        self.page_label = QLabel("Page 1")
        pagination_layout.addWidget(self.page_label)
        
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(self.next_page)
        self.next_button.setEnabled(False)
        pagination_layout.addWidget(self.next_button)
        
        results_layout.addLayout(pagination_layout)
        results_group.setLayout(results_layout)
        
        main_layout.addWidget(results_group)
        
        # Status label
        self.status_label = QLabel("Ready to search")
        main_layout.addWidget(self.status_label)
        
        # Set the main layout
        self.setLayout(main_layout)
        
        # Initial message
        self.show_initial_message()
    
    def show_initial_message(self):
        """Show initial welcome message"""
        # Clear existing results
        self.clear_results()
        
        # Create welcome widget
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout()
        
        # Icon or logo if available
        try:
            logo_label = QLabel()
            logo_pixmap = QPixmap("assets/icons/huggingface_logo.png")
            if not logo_pixmap.isNull():
                logo_label.setPixmap(logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                logo_label.setAlignment(Qt.AlignCenter)
                welcome_layout.addWidget(logo_label)
        except Exception:
            pass
        
        # Title
        title_label = QLabel("Hugging Face Model Browser")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Search for and load models compatible with your system from Hugging Face Hub.\n"
            "Use the filters above to narrow down your search results."
        )
        desc_label.setStyleSheet("font-size: 16px;")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        welcome_layout.addWidget(desc_label)
        
        # Suggestions
        suggestions_label = QLabel(
            "<b>Try searching for:</b><br>"
            "• 'code' for code generation models<br>"
            "• 'chat' for conversation models<br>"
            "• 'llama' for LLaMA-based models<br>"
            "• 'quantized' for smaller, optimized models"
        )
        suggestions_label.setStyleSheet("font-size: 14px;")
        suggestions_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(suggestions_label)
        
        welcome_widget.setLayout(welcome_layout)
        self.results_layout.addWidget(welcome_widget)
        
        # Add stretcher
        self.results_layout.addStretch()
    
    def search_models(self):
        """Search for models based on current filters"""
        # Check if already searching
        if self.is_searching:
            self.cancel_search()
            return
        
        # Update UI for search start
        self.is_searching = True
        self.search_button.setText("Cancel")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Searching for models...")
        
        # Prepare search parameters
        search_params = {
            "query": self.search_input.text().strip(),
            "page": self.current_page,
            "category": self.category_combo.currentText(),
            "size": self.size_combo.currentText(),
            "framework": self.framework_combo.currentText(),
            "license": self.license_combo.currentText(),
            "compatibility_filter": self.compat_check.isChecked(),
            "show_quantized": self.show_quantized.isChecked()
        }
        
        # Create and start search thread
        self.search_thread = ModelSearchThread(search_params)
        self.search_thread.results_ready.connect(self.handle_search_results)
        self.search_thread.error_occurred.connect(self.handle_search_error)
        self.search_thread.progress_updated.connect(self.update_progress)
        self.search_thread.start()
    
    def cancel_search(self):
        """Cancel ongoing search"""
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.cancel()
            self.search_thread.wait()
        
        self.reset_search_ui()
    
    def reset_search_ui(self):
        """Reset UI after search completion or cancellation"""
        self.is_searching = False
        self.search_button.setText("Search")
        self.progress_bar.setVisible(False)
    
    def handle_search_results(self, results, total_count):
        """Handle search results from the thread"""
        # Reset search UI
        self.reset_search_ui()
        
        # Clear existing results
        self.clear_results()
        
        # Update results count
        self.total_results = total_count
        
        if total_count == 0:
            self.results_count_label.setText("No results found")
            self.show_no_results_message()
        else:
            self.results_count_label.setText(f"Showing {len(results)} of {total_count} results")
            
            # Add results
            for model_data in results:
                model_widget = HuggingFaceModelWidget(model_data)
                model_widget.model_selected.connect(self.on_model_selected)
                self.results_layout.addWidget(model_widget)
            
            # Add stretcher
            self.results_layout.addStretch()
        
        # Update pagination
        self.update_pagination()
        
        # Update status
        self.status_label.setText(f"Found {total_count} models matching your search")
    
    def handle_search_error(self, error_message):
        """Handle search error"""
        # Reset search UI
        self.reset_search_ui()
        
        # Show error message
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.warning(
            self,
            "Search Error",
            f"An error occurred while searching for models:\n\n{error_message}"
        )
    
    def update_progress(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
    
    def clear_results(self):
        """Clear all results from the container"""
        # Remove all widgets from the layout
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def show_no_results_message(self):
        """Show message when no results are found"""
        no_results = QLabel("No models found matching your search criteria.")
        no_results.setStyleSheet("font-size: 16px; color: #6C757D;")
        no_results.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(no_results)
        
        suggestions = QLabel(
            "Suggestions:<br>"
            "• Try using fewer filters<br>"
            "• Use more general search terms<br>"
            "• Uncheck 'Show only models compatible with my system'"
        )
        suggestions.setStyleSheet("font-size: 14px;")
        suggestions.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(suggestions)
        
        # Add stretcher
        self.results_layout.addStretch()
    
    def update_pagination(self):
        """Update pagination controls"""
        # Update page label
        self.page_label.setText(f"Page {self.current_page + 1}")
        
        # Update buttons
        self.prev_button.setEnabled(self.current_page > 0)
        
        has_more = (self.current_page + 1) * MODELS_PER_PAGE < self.total_results
        self.next_button.setEnabled(has_more)
    
    def prev_page(self):
        """Go to previous page of results"""
        if self.current_page > 0:
            self.current_page -= 1
            self.search_models()
    
    def next_page(self):
        """Go to next page of results"""
        self.current_page += 1
        self.search_models()
    
    def on_model_selected(self, model_data):
        """Handle model selection"""
        model_id = model_data.get("id", "")
        
        if not model_id:
            return
        
        # Confirm loading
        reply = QMessageBox.question(
            self,
            "Load Model",
            f"Do you want to load the model '{model_data.get('modelId', model_id)}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Emit signal
            self.model_selected.emit(model_id)
            
            # Update status
            self.status_label.setText(f"Model '{model_id}' selected for loading")
    
    def show_recommendation_dialog(self):
        """Show the model recommendation dialog"""
        dialog = ModelRecommendationDialog(self)
        
        # Connect the model selected signal
        dialog.model_selected.connect(self.on_recommendation_selected)
        
        # Show the dialog
        dialog.exec_()

    def on_recommendation_selected(self, model_id):
        """Handle model selection from recommendations"""
        # Update search field
        self.search_input.setText(model_id)
        
        # Select the model
        self.model_selected.emit(model_id)


class ModelRecommendationDialog(QDialog):
    """Dialog for recommending models based on user needs"""
    
    model_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Model Recommendations")
        self.setMinimumSize(600, 400)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Preferences section
        prefs_group = QGroupBox("What kind of model are you looking for?")
        prefs_layout = QFormLayout()
        
        # Task selection
        self.task_combo = QComboBox()
        self.task_combo.addItems([
            "Code Generation",
            "Conversational Assistant",
            "Text Generation",
            "Summarization",
            "Translation"
        ])
        prefs_layout.addRow("Primary task:", self.task_combo)
        
        # Size preference
        self.size_combo = QComboBox()
        self.size_combo.addItems([
            "Smaller & Faster (less powerful)",
            "Balanced size and quality",
            "Larger & More Powerful (slower)"
        ])
        prefs_layout.addRow("Size preference:", self.size_combo)
        
        # Quality preference
        self.quality_combo = QComboBox()
        self.quality_combo.addItems([
            "Best quality (may be slower)",
            "Balanced quality and speed",
            "Fastest response (may sacrifice quality)"
        ])
        prefs_layout.addRow("Performance priority:", self.quality_combo)
        
        # Include local GPU option if available
        if torch.cuda.is_available():
            self.gpu_check = QCheckBox("Use GPU acceleration")
            self.gpu_check.setChecked(True)
            
            # Get GPU info
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
            self.gpu_check.setToolTip(f"Using {gpu_name} with {gpu_mem:.1f}GB memory")
            prefs_layout.addRow("Hardware:", self.gpu_check)
        
        # Quantization option
        self.quant_check = QCheckBox("Use quantized models (smaller & faster)")
        self.quant_check.setChecked(True)
        self.quant_check.setToolTip("Load models in 4-bit or 8-bit precision to reduce memory usage")
        prefs_layout.addRow("Optimization:", self.quant_check)
        
        prefs_group.setLayout(prefs_layout)
        layout.addWidget(prefs_group)
        
        # Recommendation button
        get_rec_button = QPushButton("Get Recommendations")
        get_rec_button.setStyleSheet("background-color: #007BFF; color: white; padding: 8px;")
        get_rec_button.clicked.connect(self.get_recommendations)
        layout.addWidget(get_rec_button)
        
        # Results area
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_widget.setLayout(self.results_layout)
        self.results_widget.setVisible(False)
        layout.addWidget(self.results_widget)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_recommendations(self):
        """Get model recommendations based on preferences"""
        # Clear existing results
        self.clear_results()
        
        # Get preferences
        task = self.task_combo.currentText()
        size_pref = self.size_combo.currentText()
        quality_pref = self.quality_combo.currentText()
        use_gpu = hasattr(self, 'gpu_check') and self.gpu_check.isChecked()
        use_quant = self.quant_check.isChecked()
        
        # Determine system capabilities
        system_capabilities = self.get_system_capabilities()
        
        # Get recommendations
        recommendations = self.recommend_models(
            task, size_pref, quality_pref, use_gpu, use_quant, system_capabilities
        )
        
        if not recommendations:
            # No recommendations (should not happen, but just in case)
            no_rec_label = QLabel("No suitable models found for your requirements.")
            self.results_layout.addWidget(no_rec_label)
        else:
            # Show recommendations header
            header_label = QLabel("Recommended Models")
            header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            self.results_layout.addWidget(header_label)
            
            # Show hardware info
            hw_info = "System capabilities: "
            if system_capabilities["has_gpu"]:
                hw_info += f"GPU with {system_capabilities['gpu_memory_gb']:.1f}GB VRAM"
                if use_gpu:
                    hw_info += " (enabled)"
                else:
                    hw_info += " (disabled)"
            else:
                hw_info += "CPU only"
            
            hw_info += f", {system_capabilities['system_memory_gb']:.1f}GB RAM"
            
            hw_label = QLabel(hw_info)
            hw_label.setStyleSheet("font-style: italic; color: #6c757d;")
            self.results_layout.addWidget(hw_label)
            
            # Add model recommendations
            for i, rec in enumerate(recommendations):
                model_frame = QFrame()
                model_frame.setFrameShape(QFrame.StyledPanel)
                model_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px;")
                
                model_layout = QVBoxLayout()
                
                # Title and description
                title_label = QLabel(f"<b>{i+1}. {rec['name']}</b>")
                title_label.setStyleSheet("font-size: 14px;")
                model_layout.addWidget(title_label)
                
                desc_label = QLabel(rec["description"])
                desc_label.setWordWrap(True)
                model_layout.addWidget(desc_label)
                
                # Tags
                tags_layout = QHBoxLayout()
                for tag in rec["tags"]:
                    tag_label = QLabel(tag)
                    tag_label.setStyleSheet(
                        "background-color: #e9ecef; padding: 3px 8px; border-radius: 10px;"
                    )
                    tags_layout.addWidget(tag_label)
                
                tags_layout.addStretch()
                model_layout.addLayout(tags_layout)
                
                # Load button
                load_button = QPushButton("Load Model")
                load_button.setProperty("model_id", rec["id"])
                load_button.clicked.connect(self.on_model_load)
                model_layout.addWidget(load_button)
                
                model_frame.setLayout(model_layout)
                self.results_layout.addWidget(model_frame)
        
        # Make results visible
        self.results_widget.setVisible(True)
        
        # Adjust dialog size
        self.adjustSize()
    
    def clear_results(self):
        """Clear all results from the container"""
        # Remove all widgets from the layout
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def get_system_capabilities(self):
        """Get system capabilities for model recommendations"""
        capabilities = {
            "has_gpu": torch.cuda.is_available(),
            "gpu_memory_gb": 0,
            "system_memory_gb": self.get_system_memory_gb(),
            "cpu_count": os.cpu_count() or 4
        }
        
        # Get GPU memory if available
        if capabilities["has_gpu"]:
            try:
                device = torch.cuda.current_device()
                capabilities["gpu_memory_gb"] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            except Exception as e:
                logger.warning(f"Error getting GPU memory: {str(e)}")
        
        return capabilities
    
    def get_system_memory_gb(self):
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback if psutil not available
            return 8.0  # Default assumption
    
    def recommend_models(self, task, size_pref, quality_pref, use_gpu, use_quant, system_caps):
        """Generate model recommendations based on preferences and system capabilities"""
        recommendations = []
        
        # Define recommendation sets based on task
        if task == "Code Generation":
            if use_gpu and system_caps["gpu_memory_gb"] >= 24 and "Larger" in size_pref:
                recommendations.append({
                    "id": "Phind/Phind-CodeLlama-34B-v2",
                    "name": "Phind CodeLlama 34B v2",
                    "description": "A powerful 34B parameter model fine-tuned for code generation with excellent performance on coding benchmarks.",
                    "tags": ["34B", "Best Quality", "GGUF"]
                })
            
            if use_gpu and system_caps["gpu_memory_gb"] >= 16:
                recommendations.append({
                    "id": "codellama/CodeLlama-13b-Instruct-hf",
                    "name": "CodeLlama 13B Instruct",
                    "description": "Meta's 13B parameter model specifically optimized for code generation tasks.",
                    "tags": ["13B", "Balanced", "Official"]
                })
            
            if use_quant:
                recommendations.append({
                    "id": "TheBloke/CodeLlama-13B-Instruct-GPTQ",
                    "name": "CodeLlama 13B Instruct (Quantized)",
                    "description": "Quantized version of CodeLlama 13B that uses significantly less memory while maintaining good performance.",
                    "tags": ["13B", "4-bit", "Memory Efficient"]
                })
            
            recommendations.append({
                "id": "Salesforce/codegen-6B-mono",
                "name": "CodeGen 6B Mono",
                "description": "Salesforce's 6B parameter model trained on code repositories across multiple programming languages.",
                "tags": ["6B", "Multi-language", "Compact"]
            })
            
            recommendations.append({
                "id": "bigcode/starcoderbase",
                "name": "StarCoderBase",
                "description": "A 15B parameter model trained on permissively licensed code. Works well for many programming languages.",
                "tags": ["15B", "Many Languages", "Popular"]
            })

            # Always include smaller options
            recommendations.append({
                "id": "replit/replit-code-v1-3b",
                "name": "Replit Code 3B",
                "description": "A compact 3B parameter model fine-tuned for code generation that works well on limited hardware.",
                "tags": ["3B", "Fast", "Low Memory"]
            })
        
        elif task == "Conversational Assistant":
            if use_gpu and system_caps["gpu_memory_gb"] >= 40 and "Larger" in size_pref:
                recommendations.append({
                    "id": "meta-llama/Llama-2-70b-chat-hf",
                    "name": "Llama 2 70B Chat",
                    "description": "Meta's largest conversation model with excellent performance across a wide range of tasks.",
                    "tags": ["70B", "Best Quality", "Official"]
                })
            
            if use_gpu and system_caps["gpu_memory_gb"] >= 16:
                recommendations.append({
                    "id": "meta-llama/Llama-2-13b-chat-hf",
                    "name": "Llama 2 13B Chat",
                    "description": "A balanced conversation model offering good performance with reasonable hardware requirements.",
                    "tags": ["13B", "Balanced", "Official"]
                })
            
            if use_quant:
                recommendations.append({
                    "id": "TheBloke/Llama-2-13B-chat-GGUF",
                    "name": "Llama 2 13B Chat (GGUF)",
                    "description": "Quantized version of Llama 2 13B Chat that runs efficiently on limited hardware.",
                    "tags": ["13B", "Memory Efficient", "Fast Loading"]
                })
            
            # Always include smaller options
            recommendations.append({
                "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "name": "TinyLlama 1.1B Chat",
                "description": "Extremely compact chat model that can run on almost any hardware with minimal requirements.",
                "tags": ["1.1B", "Very Fast", "Minimal Hardware"]
            })
        
        elif task == "Text Generation":
            if use_gpu and system_caps["gpu_memory_gb"] >= 40 and "Larger" in size_pref:
                recommendations.append({
                    "id": "databricks/dbrx-instruct",
                    "name": "DBRX Instruct",
                    "description": "A powerful instruction-tuned model that excels at creative writing and general text generation.",
                    "tags": ["50B", "Best Quality", "Creative"]
                })
            
            if use_gpu and system_caps["gpu_memory_gb"] >= 24:
                recommendations.append({
                    "id": "mistralai/Mistral-7B-Instruct-v0.2",
                    "name": "Mistral 7B Instruct v0.2",
                    "description": "A highly efficient 7B parameter model with performance rivaling much larger models.",
                    "tags": ["7B", "Efficient", "High Quality"]
                })
            
            if use_quant:
                recommendations.append({
                    "id": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                    "name": "Mistral 7B Instruct v0.2 (Quantized)",
                    "description": "Quantized version of Mistral 7B that runs efficiently on most hardware while maintaining quality.",
                    "tags": ["7B", "4-bit", "Memory Efficient"]
                })
            
            # Always include smaller options
            recommendations.append({
                "id": "EleutherAI/pythia-2.8b",
                "name": "Pythia 2.8B",
                "description": "A compact and efficient general-purpose language model suitable for most basic text generation tasks.",
                "tags": ["2.8B", "Fast", "Low Memory"]
            })
        
        # Filter based on size preference if needed
        if "Smaller" in size_pref:
            # Keep only smaller models (under 7B parameters)
            recommendations = [rec for rec in recommendations if any(tag in rec["tags"] for tag in ["1.1B", "2.8B", "3B", "6B", "Fast", "Low Memory"])]
        elif "Larger" in size_pref:
            # Prioritize larger models
            recommendations.sort(key=lambda rec: 1 if any(tag in rec["tags"] for tag in ["34B", "70B", "50B", "Best Quality"]) else 0, reverse=True)
        
        # Adjust for hardware limitations
        if not use_gpu or system_caps["gpu_memory_gb"] < 8:
            # CPU only or minimal GPU, prioritize quantized and small models
            recommendations = [rec for rec in recommendations if any(tag in rec["tags"] for tag in ["4-bit", "Memory Efficient", "Fast", "Low Memory", "Minimal Hardware"])]
        
        # Ensure we have at least some recommendations
        if not recommendations:
            # Add fallback recommendations based on task
            if task == "Code Generation":
                recommendations.append({
                    "id": "replit/replit-code-v1-3b",
                    "name": "Replit Code 3B",
                    "description": "A compact 3B parameter model fine-tuned for code generation that works well on limited hardware.",
                    "tags": ["3B", "Fast", "Low Memory"]
                })
            else:
                recommendations.append({
                    "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "name": "TinyLlama 1.1B Chat",
                    "description": "Extremely compact model that can run on almost any hardware with minimal requirements.",
                    "tags": ["1.1B", "Very Fast", "Minimal Hardware"]
                })
        
        # Limit to top recommendations
        return recommendations[:5]
    
    def on_model_load(self):
        """Handle model load button click"""
        button = self.sender()
        model_id = button.property("model_id")
        
        if model_id:
            # Emit signal
            self.model_selected.emit(model_id)
            
            # Close dialog
            self.accept()