"""
Advanced Code Generator - Main Application Module
This module defines the CodeGeneratorApp class that integrates all components.
"""

import os
import sys
import logging
import torch
import gc
from functools import partial
from typing import Optional, Tuple, Any

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QTabWidget, QAction, QMenu, QMessageBox, 
                           QStatusBar, QLabel, QStyleFactory, QApplication,
                           QSplashScreen)
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QIcon, QFont, QPixmap

# Import core functionality
from src.core.settings import SettingsManager
from src.core.model_manager import ModelManager
from src.core.error_handler import ErrorHandler

# Import GUI components
from src.gui.conversation_tab import ConversationTab
from src.gui.model_tab import ModelSelectionTab
from src.gui.code_tab import EnhancedCodeGenerationTab
from src.gui.dialogs.about_dialog import AboutDialog
from src.gui.dialogs.settings_dialog import SettingsDialog
from src.gui.widgets.memory_monitor import MemoryMonitorWidget
from src.gui.huggingface_tab import HuggingFaceTab
from src.gui.history_tab import HistoryTab
from src.gui.template_tab import TemplateTab

# Import data components
from src.data.history_manager import HistoryManager
from src.data.template_manager import TemplateManager

# Import version info
from src.version import VERSION, VERSION_DISPLAY

# Logger for this module
logger = logging.getLogger(__name__)

class CodeGeneratorApp(QMainWindow):
    """Main application window that integrates all components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the application
        
        Args:
            config_path: Optional path to configuration file
        """
        super().__init__()
        
        # Initialize settings manager
        self.settings_manager = SettingsManager(config_path)
        
        # Set up window
        self.setWindowTitle(f"Advanced Code Generator {VERSION_DISPLAY}")
        self.setGeometry(100, 100, 1280, 800)
        
        # Set window icon if available
        icon_path = os.path.join("assets", "icons", "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self)
        
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
        
        # Set up auto-save timer
        self.setup_auto_save()
        
        logger.info(f"Application initialized (v{VERSION})")
    
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
            # Create and add the tab
            self.huggingface_tab = HuggingFaceTab(model_manager=self.model_manager)
            self.tab_widget.addTab(self.huggingface_tab, "Hugging Face")
            
            # Connect signals
            self.huggingface_tab.model_selected.connect(self.on_huggingface_model_selected)
            
            logger.info("Hugging Face tab added successfully")
        except Exception as e:
            logger.error(f"Error adding Hugging Face tab: {str(e)}")
            self.error_handler.show_error("Failed to initialize Hugging Face integration", str(e))
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Set central widget
        self.setCentralWidget(main_widget)
        
        # Set up status bar
        self.setup_status_bar()
        
        # Set up menu
        self.setup_menu()
        
        # Connect tab signals
        self.connect_tab_signals()
    
    def connect_tab_signals(self):
        """Connect signals between tabs"""
        # Model tab signals
        self.model_tab.model_loaded.connect(self.on_model_loaded)
        self.model_tab.model_unloaded.connect(self.on_model_unloaded)
        
        # History tab signals
        self.history_tab.entry_selected.connect(self.on_history_entry_selected)
        
        # Template tab signals
        self.template_tab.template_applied.connect(self.on_template_applied)
    
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
        
        # New session action
        new_action = QAction("New Session", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_session)
        file_menu.addAction(new_action)
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
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
    
    def setup_auto_save(self):
        """Set up auto-save timer"""
        auto_save_interval = self.settings_manager.get_value("autosave_interval", 5)
        
        if auto_save_interval > 0:
            self.auto_save_timer = QTimer()
            self.auto_save_timer.timeout.connect(self.auto_save)
            self.auto_save_timer.start(auto_save_interval * 60 * 1000)  # Convert minutes to milliseconds
            logger.info(f"Auto-save enabled with {auto_save_interval} minute interval")
    
    def auto_save(self):
        """Perform auto-save operations"""
        try:
            # Save history
            if hasattr(self, 'history_manager'):
                self.history_manager.save_history()
            
            # Save templates
            if hasattr(self, 'template_manager'):
                self.template_manager.save_templates()
            
            # Save settings
            if hasattr(self, 'settings_manager'):
                self.settings_manager.save()
            
            logger.debug("Auto-save performed")
        except Exception as e:
            logger.error(f"Error during auto-save: {str(e)}")
    
    def new_session(self):
        """Start a new session"""
        # Check if there are unsaved changes or active processes
        if self.has_active_processes():
            confirm = QMessageBox.question(
                self,
                "New Session",
                "Starting a new session will stop all active processes. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm == QMessageBox.No:
                return
            
            # Stop active processes
            self.stop_active_processes()
        
        # Reset all tabs
        if hasattr(self.conversation_tab, 'clear_conversation'):
            self.conversation_tab.clear_conversation()
        
        if hasattr(self.code_tab, 'new_generation'):
            self.code_tab.new_generation()
        
        # Unload model
        if self.model_manager.is_model_loaded():
            self.model_tab.unload_model()
        
        # Update status
        self.status_label.setText("New session started")
        logger.info("New session started")
    
    def has_active_processes(self) -> bool:
        """Check if there are active processes running"""
        # Check code generation
        if hasattr(self.code_tab, 'code_generator') and self.code_tab.code_generator.is_generating():
            return True
        
        # Check conversation generation
        if hasattr(self.conversation_tab, 'is_generating') and self.conversation_tab.is_generating:
            return True
        
        return False
    
    def stop_active_processes(self):
        """Stop all active processes"""
        # Stop code generation
        if hasattr(self.code_tab, 'code_generator') and self.code_tab.code_generator.is_generating():
            self.code_tab.code_generator.stop()
        
        # Stop conversation generation
        if hasattr(self.conversation_tab, 'stop_generation'):
            self.conversation_tab.stop_generation()
    
    def update_memory_info(self):
        """Update memory usage information"""
        self.memory_widget.update_stats()
    
    def on_model_loaded(self, model_id: str, tokenizer: Any, model: Any):
        """
        Handle model loaded event
        
        Args:
            model_id: Model identifier
            tokenizer: Model tokenizer
            model: The loaded model
        """
        try:
            # Update code tab
            if hasattr(self.code_tab, 'set_model'):
                self.code_tab.set_model(model_id, tokenizer, model)
            
            # Update status
            self.status_label.setText(f"Model {model_id} loaded successfully")
            
            # Switch to code generation tab
            self.tab_widget.setCurrentWidget(self.code_tab)
            
            logger.info(f"Model {model_id} loaded and set in UI")
        except Exception as e:
            logger.error(f"Error handling model loaded event: {str(e)}")
            self.error_handler.show_error("Model Loaded Error", 
                                        f"Error setting up model in UI: {str(e)}")
    
    def on_model_unloaded(self):
        """Handle model unloaded event"""
        try:
            # Update code tab
            if hasattr(self.code_tab, 'set_model'):
                self.code_tab.set_model(None, None, None)
            
            # Update status
            self.status_label.setText("Model unloaded")
            
            # Force garbage collection
            self.force_memory_cleanup()
            
            logger.info("Model unloaded from UI")
        except Exception as e:
            logger.error(f"Error handling model unloaded event: {str(e)}")
    
    def on_history_entry_selected(self, entry):
        """
        Handle history entry selection
        
        Args:
            entry: The selected history entry
        """
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
            self.error_handler.show_error("History Entry Error",
                                        f"Failed to load history entry: {str(e)}")
    
    def on_template_applied(self, prompt, language):
        """
        Handle template application
        
        Args:
            prompt: The generated prompt text
            language: The selected programming language
        """
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
            self.error_handler.show_error("Template Error",
                                        f"Failed to apply template: {str(e)}")
    
    def on_huggingface_model_selected(self, model_id):
        """
        Handle model selection from the Hugging Face tab
        
        Args:
            model_id: The selected model ID
        """
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
    
    def change_theme(self, style_name):
        """
        Change application theme
        
        Args:
            style_name: The style/theme name
        """
        try:
            self.setStyle(QStyleFactory.create(style_name))
            self.setPalette(self.style().standardPalette())
            
            # Save to settings
            self.settings_manager.set_value("theme", style_name)
            
            self.status_label.setText(f"Theme changed to {style_name}")
            logger.info(f"Theme changed to {style_name}")
        except Exception as e:
            logger.error(f"Failed to change theme: {str(e)}")
            self.error_handler.show_error("Theme Error",
                                        f"Failed to change theme: {str(e)}")
    
    def change_font_size(self, size):
        """
        Change application font size
        
        Args:
            size: The font size in points
        """
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
            self.error_handler.show_error("Font Size Error",
                                        f"Failed to change font size: {str(e)}")
    
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
                if self.model_manager.is_model_loaded():
                    self.model_tab.unload_model()
                
                # Clear the model cache
                self.model_manager.clear_cache()
                
                # Force garbage collection
                self.force_memory_cleanup()
                
                self.status_label.setText("Model cache cleared successfully")
                logger.info("Model cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear model cache: {str(e)}")
                self.error_handler.show_error("Cache Error",
                                            f"Failed to clear model cache: {str(e)}")
    
    def force_memory_cleanup(self):
        """Force cleanup of unused memory"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clean up CUDA memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA memory cache emptied")
            
            # Update memory info
            self.update_memory_info()
            
            self.status_label.setText("Memory cleanup completed")
            logger.info("Manual memory cleanup performed")
        except Exception as e:
            logger.error(f"Memory cleanup failed: {str(e)}")
            self.error_handler.show_error("Memory Cleanup Error",
                                        f"Memory cleanup failed: {str(e)}")
    
    def show_about_dialog(self):
        """Show about dialog with application information"""
        about_dialog = AboutDialog(self)
        about_dialog.exec_()
    
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
        # Update theme
        theme = self.settings_manager.get_value("theme")
        if theme and theme in QStyleFactory.keys():
            self.setStyle(QStyleFactory.create(theme))
        
        # Update font size
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
        
        # Update auto-save timer
        auto_save_interval = self.settings_manager.get_value("autosave_interval", 5)
        if hasattr(self, 'auto_save_timer'):
            if auto_save_interval > 0:
                self.auto_save_timer.setInterval(auto_save_interval * 60 * 1000)
                if not self.auto_save_timer.isActive():
                    self.auto_save_timer.start()
            else:
                self.auto_save_timer.stop()
        
        # Refresh components that depend on settings
        self.update_memory_info()
        
        # Update status
        self.status_label.setText("Settings applied")
        logger.info("Settings applied")
    
    def closeEvent(self, event):
        """
        Handle application close event
        
        Args:
            event: The close event
        """
        # Check if there's an active generation
        if self.has_active_processes():
            confirm = QMessageBox.question(
                self,
                "Exit Confirmation",
                "Active processes are running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm == QMessageBox.No:
                event.ignore()
                return
            
            # Stop all active processes
            self.stop_active_processes()
        
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


def main():
    """Main entry point for the application"""
    # Create application
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("Advanced Code Generator")
    app.setOrganizationName("AdvancedCodeGenerator")
    app.setOrganizationDomain("advancedcodegenerator.ai")
    
    # Show splash screen
    splash_path = os.path.join("assets", "icons", "splash.png")
    splash_pixmap = QPixmap(splash_path)
    if splash_pixmap.isNull():
        # Fallback splash with version text
        splash_pixmap = QPixmap(400, 300)
        splash_pixmap.fill(Qt.white)
    
    splash = QSplashScreen(splash_pixmap)
    splash.showMessage(f"Starting Advanced Code Generator v{VERSION}...", 
                      Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    splash.show()
    app.processEvents()
    
    # Create main window
    main_window = CodeGeneratorApp()
    
    # Show main window and close splash
    main_window.show()
    splash.finish(main_window)
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
