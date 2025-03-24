"""
Main application integration example for the Advanced Code Generator.
This demonstrates how to connect all components.
"""

import os
import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTabWidget, QMessageBox, QSplashScreen)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QPixmap

# Import core components
from src.core.model_manager import ModelManager
from src.core.settings import SettingsManager

# Import UI components
from src.gui.model_tab import ModelSelectionTab
from src.gui.code_tab import EnhancedCodeGenerationTab
from src.gui.history_tab import HistoryTab
from src.gui.template_tab import TemplateTab

# Import data components
from src.data.history_manager import HistoryManager
from src.data.template_manager import TemplateManager

# Import version info
from src.version import VERSION

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("advanced_code_generator")

class AdvancedCodeGeneratorApp(QMainWindow):
    """Main application window for Advanced Code Generator"""
    
    def __init__(self, config_path=None):
        super().__init__()
        
        # Initialize settings manager
        self.settings_manager = SettingsManager(config_path)
        
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
        
        # Set up window
        self.setWindowTitle("Advanced Code Generator")
        self.setGeometry(100, 100, 1280, 800)
        
        # Create and set up UI
        self.setup_ui()
        
        # Load settings
        self.load_settings()
        
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
        self.code_tab = EnhancedCodeGenerationTab()
        self.history_tab = HistoryTab(self.history_manager)
        self.template_tab = TemplateTab(self.template_manager)
        
        self.tab_widget.addTab(self.model_tab, "Model Selection")
        self.tab_widget.addTab(self.code_tab, "Code Generation")
        self.tab_widget.addTab(self.history_tab, "History")
        self.tab_widget.addTab(self.template_tab, "Templates")
        
        # Connect signals
        self.connect_signals()
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Set central widget
        self.setCentralWidget(main_widget)
        
        # Set up status bar
        self.statusBar().showMessage("Ready")
    
    def connect_signals(self):
        """Connect signals between components"""
        # Model tab signals
        self.model_tab.model_loaded.connect(self.on_model_loaded)
        self.model_tab.model_unloaded.connect(self.on_model_unloaded)
        
        # History tab signals
        self.history_tab.entry_selected.connect(self.on_history_entry_selected)
        
        # Template tab signals
        self.template_tab.template_applied.connect(self.on_template_applied)
    
    def on_model_loaded(self, model_name, tokenizer, model):
        """Handle model loaded event"""
        self.code_tab.set_model(model_name, tokenizer, model)
        self.statusBar().showMessage(f"Model {model_name} loaded successfully")
        
        # Switch to code generation tab
        self.tab_widget.setCurrentWidget(self.code_tab)
    
    def on_model_unloaded(self):
        """Handle model unloaded event"""
        self.code_tab.set_model(None, None, None)
        self.statusBar().showMessage("Model unloaded")
    
    def on_history_entry_selected(self, entry):
        """Handle history entry selection"""
        # This is a placeholder for any common actions needed when a history entry is selected
        self.statusBar().showMessage(f"History entry selected: {entry.get('title', '')}")
    
    def on_template_applied(self, prompt, language):
        """Handle template application"""
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
        
        self.statusBar().showMessage("Template applied to code generator")
    
    def load_settings(self):
        """Load application settings"""
        try:
            # Window geometry
            geometry = self.settings_manager.get_value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            
            # Window state
            state = self.settings_manager.get_value("windowState")
            if state:
                self.restoreState(state)
            
            # Additional settings for components can be loaded here
            
            logger.debug("Settings loaded")
            
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Save settings
            self.settings_manager.set_value("geometry", self.saveGeometry())
            self.settings_manager.set_value("windowState", self.saveState())
            self.settings_manager.save()
            
            # Save history and templates
            self.history_manager.save_history()
            self.template_manager.save_templates()
            
            # Check if code generation is in progress
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
            
            # Unload model
            if hasattr(self.model_tab, 'unload_model'):
                self.model_tab.unload_model()
            
            logger.info("Application closing")
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during application close: {str(e)}")
            event.accept()  # Accept anyway to allow exit


if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("Advanced Code Generator")
    app.setOrganizationName("AdvancedCodeGenerator")
    app.setOrganizationDomain("advancedcodegenerator.ai")
    
    # Show splash screen
    splash_pixmap = QPixmap("assets/icons/splash.png")
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
    main_window = AdvancedCodeGeneratorApp()
    
    # Show main window and close splash
    main_window.show()
    splash.finish(main_window)
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("Advanced Code Generator")
    app.setOrganizationName("AdvancedCodeGenerator")
    app.setOrganizationDomain("advancedcodegenerator.ai")
    
    # Show splash screen
    splash_pixmap = QPixmap("assets/icons/splash.png")
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
    main_window = AdvancedCodeGeneratorApp()
    
    # Show main window and close splash
    main_window.show()
    splash.finish(main_window)
    
    # Run application
    sys.exit(app.exec_())