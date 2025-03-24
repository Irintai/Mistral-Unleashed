"""
Settings manager for the Advanced Code Generator.
Handles loading, saving, and accessing application settings.
"""

import os
import json
import yaml
import logging
from PyQt5.QtCore import QSettings

# Logger for this module
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_SETTINGS = {
    # Application settings
    "theme": "Fusion",
    "font_size": 10,
    
    # Generation settings
    "max_length": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "stream_interval": 0.05,
    
    # Model settings
    "default_model": "",
    "quantization": "8bit",
    "device": "auto",
    
    # User interface
    "show_line_numbers": True,
    "enable_code_folding": True,
    "auto_save_history": True,
    "max_history_entries": 100,
    
    # Paths
    "history_path": "",
    "templates_path": "",
    "models_cache_path": ""
}

class SettingsManager:
    """Manages application settings with layered storage"""
    
    def __init__(self, config_path=None):
        """
        Initialize the settings manager
        
        Args:
            config_path (str, optional): Path to custom configuration file
        """
        self.config_path = config_path
        
        # Qt settings for persistent storage
        self.qsettings = QSettings("AdvancedCodeGenerator", "Advanced Code Generator")
        
        # In-memory settings cache combining defaults, file config, and QSettings
        self.settings = {}
        
        # Load settings from all sources
        self.load()
        
        logger.debug("Settings manager initialized")
    
    def load(self):
        """Load settings from all sources in priority order"""
        # Start with default settings
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Load from config file if specified
        if self.config_path and os.path.exists(self.config_path):
            self.load_from_file(self.config_path)
        
        # Override with QSettings (user preferences)
        self.load_from_qsettings()
        
        # Initialize paths if they're empty
        self.initialize_paths()
        
        logger.info("Settings loaded from all sources")
    
    def load_from_file(self, file_path):
        """Load settings from config file"""
        try:
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_settings = yaml.safe_load(f)
            elif ext.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_settings = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {ext}")
                return
            
            # Update settings with file values
            if file_settings and isinstance(file_settings, dict):
                for key, value in file_settings.items():
                    self.settings[key] = value
                
                logger.info(f"Settings loaded from file: {file_path}")
            else:
                logger.warning(f"Invalid settings format in file: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading settings from file: {str(e)}")
    
    def load_from_qsettings(self):
        """Load settings from QSettings"""
        try:
            # Get all keys in QSettings
            all_keys = self.qsettings.allKeys()
            
            for key in all_keys:
                # Only override settings that exist in defaults
                if key in self.settings:
                    # Get value and convert to the same type as in defaults
                    value = self.qsettings.value(key)
                    default_value = self.settings[key]
                    
                    # Ensure correct type based on default value
                    if isinstance(default_value, bool):
                        # Handle boolean conversion specially (Qt quirk)
                        if isinstance(value, str):
                            value = value.lower() in ['true', '1', 'yes']
                        else:
                            value = bool(value)
                    elif isinstance(default_value, int):
                        value = int(value)
                    elif isinstance(default_value, float):
                        value = float(value)
                    
                    # Update settings
                    self.settings[key] = value
            
            logger.debug("Settings loaded from QSettings")
            
        except Exception as e:
            logger.error(f"Error loading settings from QSettings: {str(e)}")
    
    def initialize_paths(self):
        """Initialize default paths if not set"""
        # Get application data directory
        app_data_dir = self.get_app_data_dir()
        
        # History path
        if not self.settings["history_path"]:
            self.settings["history_path"] = os.path.join(app_data_dir, "history.json")
        
        # Templates path
        if not self.settings["templates_path"]:
            self.settings["templates_path"] = os.path.join(app_data_dir, "templates.json")
        
        # Models cache path
        if not self.settings["models_cache_path"]:
            self.settings["models_cache_path"] = os.path.join(app_data_dir, "models")
    
    def get_app_data_dir(self):
        """Get platform-specific application data directory"""
        app_name = "AdvancedCodeGenerator"
        
        if os.name == 'nt':  # Windows
            app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
            app_dir = os.path.join(app_data, app_name)
        elif os.name == 'posix':  # macOS/Linux
            if os.path.exists('/Applications'):  # macOS
                app_data = os.path.expanduser('~/Library/Application Support')
                app_dir = os.path.join(app_data, app_name)
            else:  # Linux
                app_data = os.path.expanduser('~/.local/share')
                app_dir = os.path.join(app_data, app_name)
        else:  # Fallback
            app_dir = os.path.expanduser(f'~/.{app_name.lower()}')
        
        # Create directory if it doesn't exist
        os.makedirs(app_dir, exist_ok=True)
        
        return app_dir
    
    def get_value(self, key, default=None):
        """
        Get a setting value
        
        Args:
            key (str): The setting key
            default: Default value if the key doesn't exist
            
        Returns:
            The setting value or default if not found
        """
        return self.settings.get(key, default)
    
    def set_value(self, key, value):
        """
        Set a setting value
        
        Args:
            key (str): The setting key
            value: The setting value
        """
        # Update in-memory cache
        self.settings[key] = value
        
        # Update QSettings for persistence
        try:
            self.qsettings.setValue(key, value)
        except Exception as e:
            logger.error(f"Error saving setting to QSettings: {str(e)}")
    
    def delete_value(self, key):
        """
        Delete a setting
        
        Args:
            key (str): The setting key to delete
        """
        if key in self.settings:
            del self.settings[key]
        
        # Remove from QSettings
        self.qsettings.remove(key)
    
    def get_all(self):
        """
        Get all settings
        
        Returns:
            dict: All current settings
        """
        return self.settings.copy()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        # Clear QSettings
        self.qsettings.clear()
        
        # Reset in-memory settings to defaults
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Initialize paths
        self.initialize_paths()
        
        logger.info("Settings reset to defaults")
    
    def save(self):
        """Save all current settings to QSettings"""
        try:
            # Save each setting to QSettings
            for key, value in self.settings.items():
                self.qsettings.setValue(key, value)
            
            self.qsettings.sync()
            logger.info("Settings saved to QSettings")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            return False
    
    def export_to_file(self, file_path):
        """
        Export settings to a file
        
        Args:
            file_path (str): Path to save settings to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() in ['.yml', '.yaml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.settings, f, default_flow_style=False)
            elif ext.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=2)
            else:
                logger.warning(f"Unsupported config file format: {ext}")
                return False
            
            logger.info(f"Settings exported to file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting settings to file: {str(e)}")
            return False