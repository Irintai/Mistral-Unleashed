"""
Main application class for the Advanced Code Generator.
This module defines the CodeGeneratorApp class that integrates all components.
"""

import os
import sys
import logging
import torch
import gc
from functools import partial
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QTabWidget, QAction, QMenu, QMessageBox, 
                           QStatusBar, QLabel, QStyleFactory, QApplication)
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QIcon, QFont

# Import core functionality
from src.core.settings import SettingsManager
from src.core.model_manager import ModelManager

# Import GUI components
from src.gui.conversation_tab import ConversationTab
from src.gui.model_tab import ModelSelectionTab
from src.gui.code_tab import EnhancedCodeGenerationTab
from src.gui.dialogs.about_dialog import AboutDialog
from src.gui.dialogs.settings_dialog import SettingsDialog
from src.gui.widgets.memory_monitor import MemoryMonitorWidget
from src.gui.huggingface_tab import HuggingFaceTab, ModelRecommendationDialog
from src.gui.history_tab import HistoryTab
from src.gui.template_tab import TemplateTab

# Import data components
from src.data.history_manager import HistoryManager
from src.data.template_manager import TemplateManager

# Logger for this module
logger = logging.getLogger(__name__)

class CodeGeneratorApp(QMainWindow):
    """Main application window that integrates all components"""
    
    def __init__(self, config_path=None):
        super().__init__()
        
        # Initialize settings manager
        self.settings_manager = SettingsManager(config_path)
        
        # Set up window
        self.setWindowTitle("Advanced Code Generator")
        self.setGeometry(100, 100, 1280, 800)
        
        # Set window icon if available
        icon_path = os.path.join("assets", "icons", "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize data managers
        self.history_manager = HistoryManager(
            history_path=self.settings_manager.get_value("history_path"),
            max_entries=self.settings_manager.get_value("max_history_entries", 100)
        )
        
        self.template_manager = TemplateManager(
            templates_path=self.settings_manager.get_value("templates_path")
        )
        
        # Create and set up UI
        self.setup_ui()
        
        # Load settings
        self.load_settings()
        
        # Initialize memory monitoring
        self.setup_memory_monitoring()
        
        logger.info("Application initialized")
    
    def setup_ui(self):
        """Set up the main user interface"""
        # Create central widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create and add tabs
        self.model_tab = ModelSelectionTab(self.model_manager)
        self.conversation_tab = ConversationTab(self)
        self.code_tab = EnhancedCodeGenerationTab()
        self.history_tab = HistoryTab(self.history_manager)
        self.template_tab = TemplateTab(self.template_manager)
        
        self.tab_widget.addTab(self.model_tab, "Model Selection")
        self.tab_widget.addTab(self.conversation_tab, "Conversation")
        self.tab_widget.addTab(self.code_tab, "Code Generation")
        self.tab_widget.addTab(self.history_tab, "History")
        self.tab_widget.addTab(self.template_tab, "Templates")
        
        # Add Hugging Face tab with error handling
        try:
            # Import the HuggingFaceTab class
            from src.gui.huggingface_tab import HuggingFaceTab
            
            # Create and add the tab
            self.huggingface_tab = HuggingFaceTab(model_manager=self.model_manager)
            self.tab_widget.addTab(self.huggingface_tab, "Hugging Face")
            
            # Connect signals
            self.huggingface_tab.model_selected.connect(self.on_huggingface_model_selected)
            
            print("Hugging Face tab added successfully")
        except Exception as e:
            print(f"Error adding Hugging Face tab: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Set central widget
        self.setCentralWidget(main_widget)
        
        # Set up status bar
        self.setup_status_bar()
        
        # Connect model signals to conversation tab
        self.model_tab.model_loaded.connect(
            lambda model_id, tokenizer, model: self.conversation_tab.on_model_loaded(model_id, tokenizer, model)
        )
        self.model_tab.model_unloaded.connect(self.conversation_tab.on_model_unloaded)
    
    def setup_status_bar(self):
        """Set up the application status bar"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Status message (left side)
        self.status_label = QLabel("Ready")
        self.statusBar.addWidget(self.status_label)
        
        # Memory usage display (right side)
        self.memory_widget = MemoryMonitorWidget()
        self.statusBar.addPermanentWidget(self.memory_widget)
    
    def setup_menu(self):
        """Create application menu"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Theme submenu
        theme_menu = QMenu("Themes", self)
        
        for style_name in QStyleFactory.keys():
            theme_action = QAction(style_name, self)
            theme_action.triggered.connect(partial(self.change_theme, style_name=style_name))
            theme_menu.addAction(theme_action)
        
        view_menu.addMenu(theme_menu)
        
        # Font size submenu
        font_menu = QMenu("Font Size", self)
        
        for size in [8, 9, 10, 11, 12, 14, 16]:
            font_action = QAction(f"{size}pt", self)
            font_action.triggered.connect(partial(self.change_font_size, size=size))
            font_menu.addAction(font_action)
        
        view_menu.addMenu(font_menu)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        clear_cache_action = QAction("Clear Model Cache", self)
        clear_cache_action.triggered.connect(self.clear_model_cache)
        tools_menu.addAction(clear_cache_action)
        
        memory_cleanup_action = QAction("Force Memory Cleanup", self)
        memory_cleanup_action.triggered.connect(self.force_memory_cleanup)
        tools_menu.addAction(memory_cleanup_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        model_info_action = QAction("Model Information", self)
        model_info_action.triggered.connect(self.show_model_info)
        help_menu.addAction(model_info_action)
    
    def setup_memory_monitoring(self):
        """Set up periodic memory monitoring"""
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_info)
        self.memory_timer.start(5000)  # Update every 5 seconds
    
    def update_memory_info(self):
        """Update memory usage information"""
        self.memory_widget.update_stats()
    
    def on_model_loaded(self, model_name, tokenizer, model):
        """Handle model loaded event"""
        self.code_tab.set_model(model_name, tokenizer, model)
        self.status_label.setText(f"Model {model_name} loaded successfully")
        
        # Switch to code generation tab
        self.tab_widget.setCurrentWidget(self.code_tab)
    
    def on_model_unloaded(self):
        """Handle model unloaded event"""
        self.code_tab.set_model(None, None, None)
        self.status_label.setText("Model unloaded")
    
    def on_history_entry_selected(self, entry):
        """Handle history entry selection"""
        try:
            # If the entry contains code, load it into the code generation tab
            if 'code' in entry:
                self.code_tab.code_editor.setText(entry['code'])
                
                # Set language if available
                if 'language' in entry:
                    if hasattr(self.code_tab, 'language_combo'):
                        index = self.code_tab.language_combo.findText(entry['language'])
                        if index >= 0:
                            self.code_tab.language_combo.setCurrentIndex(index)
                
                # Switch to code generation tab
                self.tab_widget.setCurrentWidget(self.code_tab)
            
            self.status_label.setText(f"Loaded history entry: {entry.get('title', '')}")
        except Exception as e:
            logger.error(f"Error loading history entry: {str(e)}")
    
    def on_template_applied(self, prompt, language):
        """Handle template application"""
        try:
            # Set prompt in code generation tab
            if hasattr(self.code_tab, 'prompt_editor'):
                self.code_tab.prompt_editor.setText(prompt)
            
            # Set language
            if hasattr(self.code_tab, 'language_combo'):
                index = self.code_tab.language_combo.findText(language)
                if index >= 0:
                    self.code_tab.language_combo.setCurrentIndex(index)
            
            # Switch to code generation tab
            self.tab_widget.setCurrentWidget(self.code_tab)
            
            self.status_label.setText("Template applied to code generator")
        except Exception as e:
            logger.error(f"Error applying template: {str(e)}")
    
    def on_huggingface_model_selected(self, model_id):
        """Handle model selection from the Hugging Face tab"""
        # Check if a model is currently loaded
        if self.model_manager.is_model_loaded():
            # Ask if user wants to unload current model first
            reply = QMessageBox.question(
                self,
                "Model Already Loaded",
                "A model is already loaded. Do you want to unload it and load the new model?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.No:
                return
            
            # Unload current model
            self.model_tab.unload_model()
        
        # Switch to model tab
        self.tab_widget.setCurrentWidget(self.model_tab)
        
        # Set the model ID in the model tab's input field
        if hasattr(self.model_tab, 'model_combo'):
            self.model_tab.model_combo.setEditText(model_id)
        
        # Update status
        self.status_label.setText(f"Model ID '{model_id}' set. Click 'Load Model' to load it.")
        
        # Optionally, you could automatically trigger model loading:
        # self.model_tab.load_model()
    
    def change_theme(self, style_name):
        """Change application theme"""
        try:
            self.setStyle(QStyleFactory.create(style_name))
            self.setPalette(self.style().standardPalette())
            
            # Save to settings
            self.settings_manager.set_value("theme", style_name)
            
            self.status_label.setText(f"Theme changed to {style_name}")
            logger.info(f"Theme changed to {style_name}")
        except Exception as e:
            logger.error(f"Failed to change theme: {str(e)}")
    
    def change_font_size(self, size):
        """Change application font size"""
        try:
            size = int(size)
            font = self.font()
            font.setPointSize(size)
            self.setFont(font)
            
            # Apply to application
            app = QApplication.instance()
            app.setFont(font)
            
            # Save to settings
            self.settings_manager.set_value("font_size", size)
            
            self.status_label.setText(f"Font size changed to {size}pt")
            logger.info(f"Font size changed to {size}pt")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to change font size: {str(e)}")
    
    def clear_model_cache(self):
        """Clear the model cache"""
        confirm = QMessageBox.question(
            self,
            "Clear Model Cache",
            "Are you sure you want to clear the model cache? "
            "This will remove all loaded models from memory.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Unload current model
                self.model_tab.unload_model()
                
                # Clear the model cache
                self.model_manager.clear_cache()
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                # Update memory info
                self.update_memory_info()
                
                self.status_label.setText("Model cache cleared successfully")
                logger.info("Model cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear model cache: {str(e)}")
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to clear model cache: {str(e)}"
                )
    
    def force_memory_cleanup(self):
        """Force cleanup of unused memory"""
        try:
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            # Update memory info
            self.update_memory_info()
            
            self.status_label.setText("Memory cleanup completed")
            logger.info("Manual memory cleanup performed")
        except Exception as e:
            logger.error(f"Memory cleanup failed: {str(e)}")
    
    def show_about_dialog(self):
        """Show about dialog with application information"""
        AboutDialog(self).exec_()
    
    def show_model_info(self):
        """Show information about the currently loaded model"""
        model_info = self.model_manager.get_current_model_info()
        if model_info:
            QMessageBox.information(
                self,
                "Model Information",
                model_info
            )
        else:
            QMessageBox.information(
                self,
                "No Model Loaded",
                "Please load a model first to view its information."
            )
    
    def show_settings_dialog(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.settings_manager, self)
        if dialog.exec_():
            # Apply settings if dialog was accepted
            self.apply_settings()
    
    def load_settings(self):
        """Load application settings"""
        # Theme
        theme = self.settings_manager.get_value("theme")
        if theme and theme in QStyleFactory.keys():
            self.setStyle(QStyleFactory.create(theme))
        
        # Font size
        font_size = self.settings_manager.get_value("font_size")
        if font_size:
            try:
                size = int(font_size)
                font = self.font()
                font.setPointSize(size)
                self.setFont(font)
                
                # Apply to application
                app = QApplication.instance()
                app.setFont(font)
            except (ValueError, TypeError):
                pass
        
        # Window geometry
        geometry = self.settings_manager.get_value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Window state
        state = self.settings_manager.get_value("windowState")
        if state:
            self.restoreState(state)
    
    def apply_settings(self):
        """Apply current settings throughout the application"""
        # This method would apply all settings that might have been changed
        # For now, we'll just log that settings were applied
        logger.info("Settings applied")
        
        # Refresh components that depend on settings
        self.update_memory_info()
        
        # Update status
        self.status_label.setText("Settings applied")
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Check if there's an active generation
        if hasattr(self.code_tab, 'code_generator') and self.code_tab.code_generator.is_generating():
            confirm = QMessageBox.question(
                self,
                "Exit Confirmation",
                "A code generation is in progress. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm == QMessageBox.No:
                event.ignore()
                return
            
            # Stop the generation
            self.code_tab.code_generator.stop_generation()
        
        # Save settings before closing
        try:
            # Window geometry
            self.settings_manager.set_value("geometry", self.saveGeometry())
            self.settings_manager.set_value("windowState", self.saveState())
            
            # Save all settings
            self.settings_manager.save()
            
            logger.info("Settings saved on exit")
        except Exception as e:
            logger.error(f"Failed to save settings: {str(e)}")
        
        # Unload any loaded models
        if hasattr(self.model_tab, 'unload_model'):
            try:
                self.model_tab.unload_model()
                logger.info("Models unloaded on exit")
            except Exception as e:
                logger.error(f"Failed to unload models: {str(e)}")
        
        # Accept the close event
        event.accept()