"""
Model selection tab for the Advanced Code Generator.
Allows users to load and manage language models.
"""

import os
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QComboBox, QPushButton, QGroupBox, QFormLayout, 
                           QRadioButton, QButtonGroup, QProgressBar, QMessageBox,
                           QSpacerItem, QSizePolicy, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont

# Logger for this module
logger = logging.getLogger(__name__)

class ModelDownloadThread(QThread):
    """Thread for downloading models in the background"""
    
    # Signals
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    download_finished = pyqtSignal(bool, str)
    
    def __init__(self, model_id, download_path, token=None, parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.download_path = download_path
        self.token = token
        self.is_stopped = False
    
    def run(self):
        """Run the model download thread"""
        try:
            # Import libraries
            from huggingface_hub import snapshot_download, HfApi
            import os
            
            # Update status
            self.status_updated.emit(f"Downloading model: {self.model_id}")
            self.progress_updated.emit(10)
            
            # Ensure download directory exists
            os.makedirs(self.download_path, exist_ok=True)
            
            # Prepare authentication
            kwargs = {
                'local_dir': self.download_path,
                'local_dir_use_symlinks': False,
                'resume_download': True,
                'ignore_patterns': ["*.pt", "*.bin"]  # Ignore large checkpoint files if needed
            }
            
            # Add token if provided
            if self.token:
                kwargs['token'] = self.token
            
            # Perform download without progress tracking
            self.status_updated.emit("Downloading model (progress updates not available)...")
            self.progress_updated.emit(30)  # Set to 30% as we begin download
            
            # Do the actual download
            snapshot_download(self.model_id, **kwargs)
            
            # Final updates
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Model {self.model_id} downloaded successfully")
            
            # Emit completion signal
            self.download_finished.emit(True, f"Model downloaded to {self.download_path}")
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            self.status_updated.emit(f"Error: {str(e)}")
            self.download_finished.emit(False, str(e))
    
    def stop(self):
        """Stop the download process"""
        self.is_stopped = True


class ModelLoadThread(QThread):
    """Thread for loading models in the background"""
    
    # Signals
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    load_finished = pyqtSignal(bool, str, object, object)  # success, message, tokenizer, model
    
    def __init__(self, model_manager, model_id, token=None, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.model_id = model_id
        self.token = token
    
    def run(self):
        """Run the model loading thread"""
        try:
            # Update status
            self.status_updated.emit(f"Preparing to load model: {self.model_id}")
            self.progress_updated.emit(10)
            
            # Set HF token if provided
            if self.token:
                self.status_updated.emit("Setting Hugging Face token")
                self.model_manager.set_huggingface_token(self.token)
            
            # Load the model
            self.status_updated.emit(f"Loading model: {self.model_id}")
            self.progress_updated.emit(30)
            
            tokenizer, model = self.model_manager.load_model(self.model_id)
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Model {self.model_id} loaded successfully")
            
            # Emit completion signal with the loaded model
            self.load_finished.emit(True, "Model loaded successfully", tokenizer, model)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            self.status_updated.emit(f"Error: {str(e)}")
            self.load_finished.emit(False, str(e), None, None)


class ModelSelectionTab(QWidget):
    """Tab for selecting and loading models"""
    
    # Signals
    model_loaded = pyqtSignal(str, object, object)  # model_id, tokenizer, model
    model_unloaded = pyqtSignal()
    
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        
        self.model_manager = model_manager
        self.load_thread = None
        self.download_thread = None
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # API token group
        token_group = QGroupBox("Hugging Face API Token")
        token_layout = QFormLayout()
        
        # Token input
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)  # Hide token
        self.token_input.setPlaceholderText("Enter your Hugging Face token (for private models)")
        token_layout.addRow("API Token:", self.token_input)
        
        # Save token checkbox
        from PyQt5.QtWidgets import QCheckBox
        self.save_token_check = QCheckBox("Remember token")
        token_layout.addRow("", self.save_token_check)
        
        token_group.setLayout(token_layout)
        layout.addWidget(token_group)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()
        
        # Model selection combobox
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.model_combo.setPlaceholderText("Enter model ID or select from list")
        
        # Add some common models to the combo box
        common_models = [
            "TheBloke/Llama-2-7B-Chat-GPTQ",
            "tiiuae/falcon-7b-instruct",
            "mosaicml/mpt-7b-instruct",
            "EleutherAI/pythia-6.9b",
            "bigcode/starcoder",
            "microsoft/CodeGPT-small-py"
        ]
        self.model_combo.addItems(common_models)
        
        model_layout.addRow("Model ID:", self.model_combo)
        
        # Quantization options
        quant_layout = QHBoxLayout()
        
        self.quant_none_radio = QRadioButton("None")
        self.quant_8bit_radio = QRadioButton("8-bit")
        self.quant_4bit_radio = QRadioButton("4-bit")
        
        self.quant_group = QButtonGroup()
        self.quant_group.addButton(self.quant_none_radio, 0)
        self.quant_group.addButton(self.quant_8bit_radio, 1)
        self.quant_group.addButton(self.quant_4bit_radio, 2)
        
        # Default to 8-bit
        self.quant_8bit_radio.setChecked(True)
        
        quant_layout.addWidget(self.quant_none_radio)
        quant_layout.addWidget(self.quant_8bit_radio)
        quant_layout.addWidget(self.quant_4bit_radio)
        
        model_layout.addRow("Quantization:", quant_layout)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "mps", "balanced", "sequential"])
        model_layout.addRow("Device:", self.device_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Load/Unload/Download actions
        actions_layout = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        actions_layout.addWidget(self.load_button)
        
        # Unload button
        self.unload_button = QPushButton("Unload Model")
        self.unload_button.clicked.connect(self.unload_model)
        self.unload_button.setEnabled(False)  # Disabled until a model is loaded
        actions_layout.addWidget(self.unload_button)
        
        # Add Download Model button
        self.download_button = QPushButton("Download Model")
        self.download_button.clicked.connect(self.download_model)
        actions_layout.addWidget(self.download_button)
        
        # Add Stop Download button
        self.stop_button = QPushButton("Stop Download")
        self.stop_button.clicked.connect(self.stop_download)
        self.stop_button.setEnabled(False)  # Disabled until a download is in progress
        actions_layout.addWidget(self.stop_button)
        
        layout.addLayout(actions_layout)
        
        # Progress components
        progress_group = QGroupBox("Loading Progress")
        progress_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Model info display
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        
        self.info_label = QLabel("No model loaded")
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Add stretcher to push all widgets to the top
        layout.addStretch()
        
        # Set the main layout
        self.setLayout(layout)
        
        # Load saved token if available
        self.load_saved_token()
    
    def download_model(self):
        """Download the selected model"""
        # Get model ID
        model_id = self.model_combo.currentText().strip()
        
        if not model_id:
            QMessageBox.warning(
                self,
                "Invalid Model ID",
                "Please enter a valid model ID or select one from the list."
            )
            return
        
        # Get token (optional)
        token = self.token_input.text().strip() or None
        
        # Choose download directory
        download_path = QFileDialog.getExistingDirectory(
            self,
            "Select Download Directory",
            os.path.expanduser("~")
        )
        
        if not download_path:
            return
        
        # Confirm download
        confirm = QMessageBox.question(
            self,
            "Confirm Download",
            f"Do you want to download the model '{model_id}' to {download_path}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.No:
            return
        
        # Disable UI during download
        self.set_ui_downloading(True)
        
        # Create and start download thread
        self.download_thread = ModelDownloadThread(
            model_id, 
            download_path, 
            token, 
            self
        )
        
        # Connect signals
        self.download_thread.progress_updated.connect(self.update_progress)
        self.download_thread.status_updated.connect(self.update_status)
        self.download_thread.download_finished.connect(self.handle_download_result)
        
        # Start download
        self.download_thread.start()
    
    def set_ui_downloading(self, is_downloading):
        """Enable/disable UI elements during downloading"""
        self.download_button.setEnabled(not is_downloading)
        self.model_combo.setEnabled(not is_downloading)
        self.token_input.setEnabled(not is_downloading)
        self.load_button.setEnabled(not is_downloading)
        
        # If downloading starts, enable stop button
        if is_downloading:
            self.stop_button.setEnabled(True)
    
    def handle_download_result(self, success, message):
        """Handle the result of model download"""
        # Re-enable UI
        self.set_ui_downloading(False)
        
        if success:
            QMessageBox.information(
                self,
                "Download Successful",
                f"Model downloaded successfully.\n\n{message}"
            )
            self.status_label.setText("Model download complete")
        else:
            QMessageBox.critical(
                self,
                "Download Failed",
                f"Failed to download model: {message}"
            )
            self.status_label.setText("Model download failed")
    
    def stop_download(self):
        """Stop ongoing model download"""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.stop()
            self.status_label.setText("Download stopped by user")
    def load_saved_token(self):
        """Load saved token from settings"""
        try:
            from PyQt5.QtCore import QSettings
            settings = QSettings("AdvancedCodeGenerator", "Advanced Code Generator")
            
            # Load saved token if available and checkbox is checked
            saved_token = settings.value("huggingface_token", "")
            token_saved = settings.value("save_token", False, type=bool)
            
            if saved_token and token_saved:
                self.token_input.setText(saved_token)
                self.save_token_check.setChecked(True)
        except Exception as e:
            logger.warning(f"Failed to load saved token: {str(e)}")
    
    def save_token(self, token):
        """Save token to settings if checkbox is checked"""
        try:
            if self.save_token_check.isChecked():
                from PyQt5.QtCore import QSettings
                settings = QSettings("AdvancedCodeGenerator", "Advanced Code Generator")
                
                settings.setValue("huggingface_token", token)
                settings.setValue("save_token", True)
                logger.debug("Token saved to settings")
            else:
                # Clear saved token if checkbox is unchecked
                from PyQt5.QtCore import QSettings
                settings = QSettings("AdvancedCodeGenerator", "Advanced Code Generator")
                
                settings.setValue("huggingface_token", "")
                settings.setValue("save_token", False)
                logger.debug("Token removed from settings")
        except Exception as e:
            logger.warning(f"Failed to save token: {str(e)}")
    
    def load_model(self):
        """Load the selected model"""
        # Get model ID
        model_id = self.model_combo.currentText().strip()
        
        if not model_id:
            QMessageBox.warning(
                self,
                "Invalid Model ID",
                "Please enter a valid model ID or select one from the list."
            )
            return
        
        # Get token
        token = self.token_input.text().strip() or None
        
        # Save token if checkbox is checked
        if token:
            self.save_token(token)
        
        # Set quantization options
        quant_id = self.quant_group.checkedId()
        if quant_id == 0:
            self.model_manager.set_quantization(False, False)
        elif quant_id == 1:
            self.model_manager.set_quantization(True, False)
        elif quant_id == 2:
            self.model_manager.set_quantization(False, True)
        
        # Set device map
        device_map = self.device_combo.currentText()
        self.model_manager.set_device_map(device_map)
        
        # Disable UI during loading
        self.set_ui_loading(True)
        
        # Create and start loading thread
        self.load_thread = ModelLoadThread(self.model_manager, model_id, token, self)
        self.load_thread.progress_updated.connect(self.update_progress)
        self.load_thread.status_updated.connect(self.update_status)
        self.load_thread.load_finished.connect(self.handle_load_result)
        self.load_thread.start()
    
    def unload_model(self):
        """Unload the current model"""
        # Check if a model is loaded
        if not self.model_manager.is_model_loaded():
            QMessageBox.information(
                self,
                "No Model Loaded",
                "There is no model currently loaded."
            )
            return
        
        try:
            # Unload the model
            success = self.model_manager.unload_model()
            
            if success:
                # Update UI
                self.status_label.setText("Model unloaded")
                self.progress_bar.setValue(0)
                self.info_label.setText("No model loaded")
                self.unload_button.setEnabled(False)
                self.load_button.setEnabled(True)
                
                # Emit signal
                self.model_unloaded.emit()
                
                # Force garbage collection
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                
                # Update memory stats display (if available)
                if hasattr(self.parent(), "memory_widget"):
                    self.parent().memory_widget.update_stats()
            else:
                QMessageBox.warning(
                    self,
                    "Unload Failed",
                    "Failed to unload the model. See log for details."
                )
                
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error unloading model: {str(e)}"
            )
    
    @pyqtSlot(int)
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
    
    @pyqtSlot(str)
    def update_status(self, message):
        """Update the status label"""
        self.status_label.setText(message)
    
    @pyqtSlot(bool, str, object, object)
    def handle_load_result(self, success, message, tokenizer, model):
        """Handle the result of model loading"""
        # Re-enable UI
        self.set_ui_loading(False)
        
        if success:
            # Update UI
            self.unload_button.setEnabled(True)
            
            # Get model info
            model_id = self.model_combo.currentText().strip()
            info_html = self.model_manager.get_current_model_info()
            self.info_label.setText(info_html)
            
            # Emit signal with loaded model
            self.model_loaded.emit(model_id, tokenizer, model)
            
            # Show success message
            QMessageBox.information(
                self,
                "Model Loaded",
                f"Model {model_id} loaded successfully."
            )
        else:
            # Show error message
            QMessageBox.critical(
                self,
                "Load Failed",
                message
            )
    
    def set_ui_loading(self, is_loading):
        """Enable/disable UI elements during loading"""
        self.token_input.setEnabled(not is_loading)
        self.model_combo.setEnabled(not is_loading)
        self.quant_none_radio.setEnabled(not is_loading)
        self.quant_8bit_radio.setEnabled(not is_loading)
        self.quant_4bit_radio.setEnabled(not is_loading)
        self.device_combo.setEnabled(not is_loading)
        self.load_button.setEnabled(not is_loading)
        
        # Only enable unload button if it was already enabled and we're disabling loading UI
        if not is_loading:
            self.unload_button.setEnabled(self.model_manager.is_model_loaded())
        else:
            self.unload_button.setEnabled(False)