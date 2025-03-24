"""
Settings dialog for configuring application parameters.
"""

import os
import logging
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
                           QWidget, QFormLayout, QLabel, QLineEdit, 
                           QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                           QPushButton, QFileDialog, QDialogButtonBox,
                           QGroupBox, QRadioButton, QButtonGroup, QScrollArea)
from PyQt5.QtCore import Qt, QSettings

# Initialize logger
logger = logging.getLogger(__name__)

class SettingsDialog(QDialog):
    """Dialog for configuring application settings"""
    
    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        
        self.settings_manager = settings_manager
        
        self.setWindowTitle("Application Settings")
        self.setMinimumSize(600, 500)
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Tab widget for settings categories
        self.tab_widget = QTabWidget()
        
        # General settings tab
        general_tab = QScrollArea()
        general_tab.setWidgetResizable(True)
        general_widget = QWidget()
        general_layout = QVBoxLayout(general_widget)
        
        # Application settings group
        app_group = QGroupBox("Application")
        app_layout = QFormLayout()
        
        # Theme selection
        self.theme_combo = QComboBox()
        from PyQt5.QtWidgets import QStyleFactory
        self.theme_combo.addItems(QStyleFactory.keys())
        app_layout.addRow("Theme:", self.theme_combo)
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setValue(10)
        app_layout.addRow("Font Size:", self.font_size_spin)
        
        # Auto-save interval
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(0, 60)
        self.autosave_spin.setValue(5)
        self.autosave_spin.setSuffix(" min")
        self.autosave_spin.setSpecialValueText("Disabled")
        app_layout.addRow("Auto-save Interval:", self.autosave_spin)
        
        app_group.setLayout(app_layout)
        general_layout.addWidget(app_group)
        
        # Editor settings group
        editor_group = QGroupBox("Code Editor")
        editor_layout = QFormLayout()
        
        # Line numbers
        self.line_numbers_check = QCheckBox("Show Line Numbers")
        editor_layout.addRow(self.line_numbers_check)
        
        # Code folding
        self.code_folding_check = QCheckBox("Enable Code Folding")
        editor_layout.addRow(self.code_folding_check)
        
        # Auto-indentation
        self.auto_indent_check = QCheckBox("Enable Auto-Indentation")
        editor_layout.addRow(self.auto_indent_check)
        
        # Tab width
        self.tab_width_spin = QSpinBox()
        self.tab_width_spin.setRange(2, 8)
        self.tab_width_spin.setValue(4)
        editor_layout.addRow("Tab Width:", self.tab_width_spin)
        
        editor_group.setLayout(editor_layout)
        general_layout.addWidget(editor_group)
        
        # History settings group
        history_group = QGroupBox("History")
        history_layout = QFormLayout()
        
        # Auto-save history
        self.save_history_check = QCheckBox("Automatically Save to History")
        history_layout.addRow(self.save_history_check)
        
        # Max history entries
        self.max_history_spin = QSpinBox()
        self.max_history_spin.setRange(10, 1000)
        self.max_history_spin.setValue(100)
        self.max_history_spin.setSingleStep(10)
        history_layout.addRow("Maximum History Entries:", self.max_history_spin)
        
        # History path
        history_path_layout = QHBoxLayout()
        self.history_path_edit = QLineEdit()
        self.history_path_edit.setReadOnly(True)
        history_path_layout.addWidget(self.history_path_edit)
        
        history_path_button = QPushButton("Browse...")
        history_path_button.clicked.connect(lambda: self.browse_path(self.history_path_edit, "Select History File Location"))
        history_path_layout.addWidget(history_path_button)
        
        history_layout.addRow("History File:", history_path_layout)
        
        history_group.setLayout(history_layout)
        general_layout.addWidget(history_group)
        
        # Add stretcher to push all widgets to the top
        general_layout.addStretch()
        
        general_tab.setWidget(general_widget)
        self.tab_widget.addTab(general_tab, "General")
        
        # Model settings tab
        model_tab = QScrollArea()
        model_tab.setWidgetResizable(True)
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        
        # Model preferences group
        model_prefs_group = QGroupBox("Model Preferences")
        model_prefs_layout = QFormLayout()
        
        # Default model
        self.default_model_edit = QLineEdit()
        model_prefs_layout.addRow("Default Model:", self.default_model_edit)
        
        # Device mapping
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "mps", "balanced", "sequential"])
        model_prefs_layout.addRow("Device Mapping:", self.device_combo)
        
        # Quantization
        quant_layout = QHBoxLayout()
        
        self.quant_none_radio = QRadioButton("None")
        self.quant_8bit_radio = QRadioButton("8-bit")
        self.quant_4bit_radio = QRadioButton("4-bit")
        
        self.quant_group = QButtonGroup()
        self.quant_group.addButton(self.quant_none_radio, 0)
        self.quant_group.addButton(self.quant_8bit_radio, 1)
        self.quant_group.addButton(self.quant_4bit_radio, 2)
        
        quant_layout.addWidget(self.quant_none_radio)
        quant_layout.addWidget(self.quant_8bit_radio)
        quant_layout.addWidget(self.quant_4bit_radio)
        
        model_prefs_layout.addRow("Quantization:", quant_layout)
        
        model_prefs_group.setLayout(model_prefs_layout)
        model_layout.addWidget(model_prefs_group)
        
        # Generation parameters group
        gen_params_group = QGroupBox("Generation Parameters")
        gen_params_layout = QFormLayout()
        
        # Max length
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(50, 2000)
        self.max_length_spin.setValue(500)
        self.max_length_spin.setSingleStep(50)
        gen_params_layout.addRow("Max Length:", self.max_length_spin)
        
        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setValue(0.7)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)
        gen_params_layout.addRow("Temperature:", self.temperature_spin)
        
        # Top-p
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.1, 1.0)
        self.top_p_spin.setValue(0.9)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setDecimals(2)
        gen_params_layout.addRow("Top-p:", self.top_p_spin)
        
        # Repetition penalty
        self.rep_penalty_spin = QDoubleSpinBox()
        self.rep_penalty_spin.setRange(1.0, 2.0)
        self.rep_penalty_spin.setValue(1.1)
        self.rep_penalty_spin.setSingleStep(0.05)
        self.rep_penalty_spin.setDecimals(2)
        gen_params_layout.addRow("Repetition Penalty:", self.rep_penalty_spin)
        
        # Stream interval
        self.stream_interval_spin = QDoubleSpinBox()
        self.stream_interval_spin.setRange(0.01, 0.5)
        self.stream_interval_spin.setValue(0.05)
        self.stream_interval_spin.setSingleStep(0.01)
        self.stream_interval_spin.setDecimals(2)
        self.stream_interval_spin.setSuffix(" sec")
        gen_params_layout.addRow("Stream Interval:", self.stream_interval_spin)
        
        gen_params_group.setLayout(gen_params_layout)
        model_layout.addWidget(gen_params_group)
        
        # Model cache group
        cache_group = QGroupBox("Model Cache")
        cache_layout = QFormLayout()
        
        # Cache path
        cache_path_layout = QHBoxLayout()
        self.cache_path_edit = QLineEdit()
        self.cache_path_edit.setReadOnly(True)
        cache_path_layout.addWidget(self.cache_path_edit)
        
        cache_path_button = QPushButton("Browse...")
        cache_path_button.clicked.connect(lambda: self.browse_directory(self.cache_path_edit, "Select Model Cache Directory"))
        cache_path_layout.addWidget(cache_path_button)
        
        cache_layout.addRow("Cache Directory:", cache_path_layout)
        
        # Clear cache button
        clear_cache_button = QPushButton("Clear Cache")
        clear_cache_button.clicked.connect(self.clear_cache)
        cache_layout.addRow("", clear_cache_button)
        
        cache_group.setLayout(cache_layout)
        model_layout.addWidget(cache_group)
        
        # Add stretcher to push all widgets to the top
        model_layout.addStretch()
        
        model_tab.setWidget(model_widget)
        self.tab_widget.addTab(model_tab, "Models")
        
        # Path settings tab
        paths_tab = QScrollArea()
        paths_tab.setWidgetResizable(True)
        paths_widget = QWidget()
        paths_layout = QVBoxLayout(paths_widget)
        
        # Templates path
        templates_group = QGroupBox("Templates")
        templates_layout = QFormLayout()
        
        templates_path_layout = QHBoxLayout()
        self.templates_path_edit = QLineEdit()
        self.templates_path_edit.setReadOnly(True)
        templates_path_layout.addWidget(self.templates_path_edit)
        
        templates_path_button = QPushButton("Browse...")
        templates_path_button.clicked.connect(lambda: self.browse_path(self.templates_path_edit, "Select Templates File Location"))
        templates_path_layout.addWidget(templates_path_button)
        
        templates_layout.addRow("Templates File:", templates_path_layout)
        templates_group.setLayout(templates_layout)
        paths_layout.addWidget(templates_group)
        
        # Custom config file
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout()
        
        # Export configuration
        export_config_layout = QHBoxLayout()
        export_config_button = QPushButton("Export Configuration...")
        export_config_button.clicked.connect(self.export_config)
        export_config_layout.addWidget(export_config_button)
        
        import_config_button = QPushButton("Import Configuration...")
        import_config_button.clicked.connect(self.import_config)
        export_config_layout.addWidget(import_config_button)
        
        config_layout.addRow("Config File:", export_config_layout)
        
        # Reset settings button
        reset_button = QPushButton("Reset All Settings")
        reset_button.clicked.connect(self.reset_settings)
        config_layout.addRow("", reset_button)
        
        config_group.setLayout(config_layout)
        paths_layout.addWidget(config_group)
        
        # Add stretcher to push all widgets to the top
        paths_layout.addStretch()
        
        paths_tab.setWidget(paths_widget)
        self.tab_widget.addTab(paths_tab, "Paths & Config")
        
        layout.addWidget(self.tab_widget)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def load_settings(self):
        """Load current settings into the dialog"""
        try:
            settings = self.settings_manager.get_all()
            
            # General settings
            self.theme_combo.setCurrentText(settings.get("theme", "Fusion"))
            self.font_size_spin.setValue(int(settings.get("font_size", 10)))
            self.autosave_spin.setValue(int(settings.get("autosave_interval", 5)))
            
            # Editor settings
            self.line_numbers_check.setChecked(settings.get("show_line_numbers", True))
            self.code_folding_check.setChecked(settings.get("enable_code_folding", True))
            self.auto_indent_check.setChecked(settings.get("auto_indent", True))
            self.tab_width_spin.setValue(int(settings.get("tab_width", 4)))
            
            # History settings
            self.save_history_check.setChecked(settings.get("auto_save_history", True))
            self.max_history_spin.setValue(int(settings.get("max_history_entries", 100)))
            self.history_path_edit.setText(settings.get("history_path", ""))
            
            # Model settings
            self.default_model_edit.setText(settings.get("default_model", ""))
            self.device_combo.setCurrentText(settings.get("device", "auto"))
            
            # Quantization setting
            quant = settings.get("quantization", "8bit")
            if quant == "none":
                self.quant_none_radio.setChecked(True)
            elif quant == "4bit":
                self.quant_4bit_radio.setChecked(True)
            else:  # Default to 8bit
                self.quant_8bit_radio.setChecked(True)
            
            # Generation parameters
            self.max_length_spin.setValue(int(settings.get("max_length", 500)))
            self.temperature_spin.setValue(float(settings.get("temperature", 0.7)))
            self.top_p_spin.setValue(float(settings.get("top_p", 0.9)))
            self.rep_penalty_spin.setValue(float(settings.get("repetition_penalty", 1.1)))
            self.stream_interval_spin.setValue(float(settings.get("stream_interval", 0.05)))
            
            # Cache path
            self.cache_path_edit.setText(settings.get("models_cache_path", ""))
            
            # Templates path
            self.templates_path_edit.setText(settings.get("templates_path", ""))
            
            logger.debug("Settings loaded into dialog")
            
        except Exception as e:
            logger.error(f"Error loading settings into dialog: {str(e)}")
    
    def apply_settings(self):
        """Apply settings from the dialog to the settings manager"""
        try:
            # General settings
            self.settings_manager.set_value("theme", self.theme_combo.currentText())
            self.settings_manager.set_value("font_size", self.font_size_spin.value())
            self.settings_manager.set_value("autosave_interval", self.autosave_spin.value())
            
            # Editor settings
            self.settings_manager.set_value("show_line_numbers", self.line_numbers_check.isChecked())
            self.settings_manager.set_value("enable_code_folding", self.code_folding_check.isChecked())
            self.settings_manager.set_value("auto_indent", self.auto_indent_check.isChecked())
            self.settings_manager.set_value("tab_width", self.tab_width_spin.value())
            
            # History settings
            self.settings_manager.set_value("auto_save_history", self.save_history_check.isChecked())
            self.settings_manager.set_value("max_history_entries", self.max_history_spin.value())
            self.settings_manager.set_value("history_path", self.history_path_edit.text())
            
            # Model settings
            self.settings_manager.set_value("default_model", self.default_model_edit.text())
            self.settings_manager.set_value("device", self.device_combo.currentText())
            
            # Quantization setting
            quant_id = self.quant_group.checkedId()
            if quant_id == 0:
                self.settings_manager.set_value("quantization", "none")
            elif quant_id == 2:
                self.settings_manager.set_value("quantization", "4bit")
            else:
                self.settings_manager.set_value("quantization", "8bit")
            
            # Generation parameters
            self.settings_manager.set_value("max_length", self.max_length_spin.value())
            self.settings_manager.set_value("temperature", self.temperature_spin.value())
            self.settings_manager.set_value("top_p", self.top_p_spin.value())
            self.settings_manager.set_value("repetition_penalty", self.rep_penalty_spin.value())
            self.settings_manager.set_value("stream_interval", self.stream_interval_spin.value())
            
            # Cache path
            self.settings_manager.set_value("models_cache_path", self.cache_path_edit.text())
            
            # Templates path
            self.settings_manager.set_value("templates_path", self.templates_path_edit.text())
            
            # Save settings
            self.settings_manager.save()
            
            logger.info("Settings applied")
            
        except Exception as e:
            logger.error(f"Error applying settings: {str(e)}")
    
    def accept(self):
        """Handle dialog acceptance"""
        self.apply_settings()
        super().accept()
    
    def browse_path(self, line_edit, title):
        """Browse for a file path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            line_edit.text() or os.path.expanduser("~"),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            line_edit.setText(file_path)
    
    def browse_directory(self, line_edit, title):
        """Browse for a directory path"""
        directory = QFileDialog.getExistingDirectory(
            self,
            title,
            line_edit.text() or os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            line_edit.setText(directory)
    
    def export_config(self):
        """Export current settings to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration",
            os.path.expanduser("~"),
            "JSON Files (*.json);;YAML Files (*.yaml);;All Files (*)"
        )
        
        if file_path:
            try:
                # Apply current settings first
                self.apply_settings()
                
                # Export settings
                result = self.settings_manager.export_to_file(file_path)
                
                if result:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Settings exported to {file_path}"
                    )
                else:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Failed to export settings"
                    )
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting settings: {str(e)}"
                )
    
    def import_config(self):
        """Import settings from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Configuration",
            os.path.expanduser("~"),
            "JSON Files (*.json);;YAML Files (*.yaml);;All Files (*)"
        )
        
        if file_path:
            try:
                # Import settings
                self.settings_manager.load_from_file(file_path)
                
                # Refresh dialog
                self.load_settings()
                
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Settings imported from {file_path}"
                )
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Error importing settings: {str(e)}"
                )
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        from PyQt5.QtWidgets import QMessageBox
        confirm = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Reset settings
                self.settings_manager.reset_to_defaults()
                
                # Refresh dialog
                self.load_settings()
                
                QMessageBox.information(
                    self,
                    "Reset Successful",
                    "Settings reset to defaults"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Reset Error",
                    f"Error resetting settings: {str(e)}"
                )
    
    def clear_cache(self):
        """Clear the model cache directory"""
        from PyQt5.QtWidgets import QMessageBox
        confirm = QMessageBox.question(
            self,
            "Clear Cache",
            "Are you sure you want to clear the model cache? This will delete all cached models.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                cache_dir = self.cache_path_edit.text()
                
                if cache_dir and os.path.exists(cache_dir):
                    import shutil
                    
                    # Delete all files in the directory
                    for filename in os.listdir(cache_dir):
                        file_path = os.path.join(cache_dir, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {str(e)}")
                    
                    QMessageBox.information(
                        self,
                        "Cache Cleared",
                        "Model cache directory has been cleared"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid Cache Directory",
                        "The cache directory does not exist"
                    )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error clearing cache: {str(e)}"
                )