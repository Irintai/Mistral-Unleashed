#!/usr/bin/env python
"""Advanced Code Generator - Launch Script
This script sets up the environment and launches the application.
"""

import os
import sys
import logging
import time
import torch
from PyQt5.QtWidgets import (QMessageBox)  
from PyQt5.QtWidgets import QApplication
from src.app import CodeGeneratorApp
from main import AdvancedCodeGeneratorApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("advanced_code_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run")

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import version info
from src.version import VERSION

def check_environment():
    """Check if the environment has all required dependencies"""
    missing_packages = []
    
    # Check core dependencies
    dependencies = [
        ("PyQt5", "PyQt5"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("yaml", "pyyaml")
    ]
    
    for module, package in dependencies:
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    # Check QsciScintilla separately as it's in a sub-package
    try:
        from PyQt5.Qsci import QsciScintilla
    except ImportError:
        missing_packages.append("PyQt5-QScintilla")
    
    if missing_packages:
        logger.error(f"Missing required dependencies: {', '.join(missing_packages)}")
        logger.error("Please install them with pip: pip install " + " ".join(missing_packages))
        return False
    else:
        return True

def main():
    """Main entry point"""
    # Check environment
    if not check_environment():
        return 1
    
    # Import PyQt5
    from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import Qt
    
    try:
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
        
        # Import the app
        logger.info("Importing main application...")
        from src.app import CodeGeneratorApp
        
        # Create main window
        logger.info("Creating main window...")
        main_window = CodeGeneratorApp()
        
        # Show main window and close splash
        main_window.show()
        splash.finish(main_window)
        
        # Run application
        logger.info("Application started successfully")
        return app.exec_()
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
    
    # Show error in GUI if possible
    try:
        QMessageBox.critical(
            None,
            "Error Starting Application",
            f"An error occurred while starting the application:\n\n{str(e)}\n\nSee log for details."
        )
    except Exception as gui_error:
        logger.error(f"Error displaying error message: {str(gui_error)}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
