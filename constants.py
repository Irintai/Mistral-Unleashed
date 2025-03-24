import os
import platform

from PyQt5.QtCore import Qt


"""
Constants used throughout the Advanced Code Generator application.
"""

# Application information
APP_NAME = "Advanced Code Generator"
APP_ORG = "AdvancedCodeGenerator"
APP_DOMAIN = "advancedcodegenerator.ai"

# File extensions for different programming languages
FILE_EXTENSIONS = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "rust": ".rs",
    "go": ".go",
    "java": ".java",
    "c++": ".cpp",
    "c#": ".cs",
    "php": ".php",
    "html": ".html",
    "css": ".css",
    "json": ".json",
    "yaml": ".yaml",
    "xml": ".xml",
    "markdown": ".md",
    "text": ".txt"
}

# Supported programming languages
SUPPORTED_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "java",
    "c++",
    "c#",
    "php",
    "html",
    "css"
]

# Comment symbols for different languages
COMMENT_SYMBOLS = {
    "python": "#",
    "javascript": "//",
    "typescript": "//",
    "rust": "//",
    "go": "//",
    "java": "//",
    "c++": "//",
    "c#": "//",
    "php": "//",
    "html": "<!-- -->",
    "css": "/* */",
    "sql": "--",
    "bash": "#",
    "ruby": "#"
}

# Default generation parameters
DEFAULT_MAX_LENGTH = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_STREAM_INTERVAL = 0.05

# Default code editor settings
DEFAULT_TAB_WIDTH = 4
DEFAULT_SHOW_LINE_NUMBERS = True
DEFAULT_ENABLE_CODE_FOLDING = True
DEFAULT_AUTO_INDENT = True
DEFAULT_FONT_SIZE = 10
DEFAULT_HIGHLIGHT_CURRENT_LINE = True

# Maximum history entries
MAX_HISTORY_ENTRIES = 100

# Resource paths
ICONS_PATH = "assets/icons"
THEMES_PATH = "assets/themes"
STYLES_PATH = "assets/styles"

# Model-related constants
DEFAULT_MODEL_TYPE = "text-generation"
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_QUANTIZATION = "8bit"  # "none", "8bit", "4bit"

# Template categories
DEFAULT_TEMPLATE_CATEGORIES = [
    "General",
    "Functions",
    "Classes",
    "Web",
    "Data",
    "Testing",
    "Algorithms",
    "UI/UX",
    "API",
    "Database",
    "Utilities",
    "Documentation"
]

# Default prompt templates
DEFAULT_PROMPT_TEMPLATES = [
    {
        "name": "Simple Function",
        "template": "Create a function in {{language}} that {{task}}.\n\nFunction signature: {{signature}}\n\nRequirements:\n- {{requirements}}",
        "language": "python",
        "category": "Functions",
        "placeholders": [
            {"name": "language", "description": "Programming language", "default": "Python"},
            {"name": "task", "description": "What the function should do", "default": "performs a specific task"},
            {"name": "signature", "description": "Function signature", "default": "def function_name(param1, param2)"},
            {"name": "requirements", "description": "Additional requirements", "default": "Include proper error handling and documentation"}
        ]
    },
    {
        "name": "Class Definition",
        "template": "Create a class in {{language}} named {{class_name}} that {{purpose}}.\n\nThe class should have these properties: {{properties}}\n\nAnd these methods: {{methods}}\n\nImplementation notes: {{notes}}",
        "language": "python",
        "category": "Classes",
        "placeholders": [
            {"name": "language", "description": "Programming language", "default": "Python"},
            {"name": "class_name", "description": "Name of the class", "default": "MyClass"},
            {"name": "purpose", "description": "Purpose of the class", "default": "manages data and operations for a specific entity"},
            {"name": "properties", "description": "Class properties/attributes", "default": "id, name, created_at"},
            {"name": "methods", "description": "Class methods", "default": "initialize, validate, save, to_dict"},
            {"name": "notes", "description": "Implementation notes", "default": "Ensure thread safety and proper validation"}
        ]
    },
    {
        "name": "API Endpoint",
        "template": "Create a {{framework}} API endpoint that {{action}} {{resource}}.\n\nEndpoint details:\n- HTTP Method: {{method}}\n- Route: {{route}}\n- Request parameters: {{params}}\n- Response format: {{response}}\n\nInclude proper error handling for: {{errors}}",
        "language": "python",
        "category": "API",
        "placeholders": [
            {"name": "framework", "description": "Web framework", "default": "Flask"},
            {"name": "action", "description": "Action performed", "default": "retrieves"},
            {"name": "resource", "description": "API resource", "default": "user data"},
            {"name": "method", "description": "HTTP method", "default": "GET"},
            {"name": "route", "description": "API route", "default": "/api/users/<id>"},
            {"name": "params", "description": "Request parameters", "default": "id (path), fields (query)"},
            {"name": "response", "description": "Response format", "default": "JSON with user data"},
            {"name": "errors", "description": "Error cases", "default": "not found, unauthorized, validation errors"}
        ]
    }
]

# UI-related constants
UI_DEFAULT_THEME = "Fusion"
UI_DEFAULT_FONT_FAMILY = "Consolas" if os.name == "nt" else "Menlo"  # Windows vs macOS/Linux
UI_DEFAULT_FONT_SIZE = 10
UI_WINDOW_DEFAULT_WIDTH = 1280
UI_WINDOW_DEFAULT_HEIGHT = 800
UI_SPLITTER_RATIO = [40, 60]  # Default ratio for top/bottom split
UI_ANIMATION_DURATION = 300  # Animation duration in milliseconds
UI_STATUSBAR_TIMEOUT = 5000  # Status bar message timeout in milliseconds

# Memory management
MEMORY_UPDATE_INTERVAL = 5000  # Memory monitor update interval in milliseconds
MEMORY_WARNING_THRESHOLD = 80  # Memory usage warning threshold (percentage)
MEMORY_CRITICAL_THRESHOLD = 90  # Memory usage critical threshold (percentage)

# Shortcut keys (as strings for use in QKeySequence)
SHORTCUT_GENERATE = "Ctrl+G"
SHORTCUT_STOP = "Ctrl+C"
SHORTCUT_SAVE = "Ctrl+S"
SHORTCUT_LOAD = "Ctrl+O"
SHORTCUT_NEW = "Ctrl+N"
SHORTCUT_COPY = "Ctrl+Shift+C"
SHORTCUT_SETTINGS = "Ctrl+,"
SHORTCUT_HELP = "F1"
SHORTCUT_QUIT = "Ctrl+Q"

# File filters for open/save dialogs
FILE_FILTER_CODE = "Code Files (*.py *.js *.ts *.rs *.go *.java *.cpp *.cs *.php *.html *.css);;All Files (*)"
FILE_FILTER_TEXT = "Text Files (*.txt);;All Files (*)"
FILE_FILTER_JSON = "JSON Files (*.json);;All Files (*)"
FILE_FILTER_CONFIG = "Configuration Files (*.json *.yaml *.yml);;All Files (*)"

# Network-related constants
NETWORK_TIMEOUT = 30  # Network timeout in seconds
HUGGINGFACE_API_BASE = "https://huggingface.co/api"
NETWORK_RETRY_COUNT = 3  # Number of retries for network operations
NETWORK_RETRY_DELAY = 1  # Delay between retries in seconds

# Model download chunk size (in bytes)
MODEL_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = "advanced_code_generator.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Temporary directory name
TEMP_DIR = "temp"

# Metrics tracking
TRACK_GENERATION_TIME = True  # Whether to track generation time
TRACK_MEMORY_USAGE = True  # Whether to track memory usage
TRACK_TOKEN_COUNT = True  # Whether to track token count

# Feature flags (for experimental features)
FEATURE_STREAMING_GENERATION = True
FEATURE_MODEL_QUANTIZATION = True
FEATURE_CUSTOM_TEMPLATES = True
FEATURE_HISTORY_SEARCH = True
FEATURE_ENHANCED_EDITOR = True
FEATURE_MEMORY_MONITOR = True

# Error messages
ERROR_NO_MODEL = "No model loaded. Please load a model first."
ERROR_GENERATION_FAILED = "Code generation failed. Please try again."
ERROR_MODEL_LOAD_FAILED = "Model loading failed. Please check your settings and try again."
ERROR_INVALID_PROMPT = "Invalid prompt. Please provide a valid prompt."
ERROR_OUT_OF_MEMORY = "Out of memory. Please try with a smaller model or enable quantization."

# Success messages
SUCCESS_MODEL_LOADED = "Model loaded successfully."
SUCCESS_CODE_GENERATED = "Code generated successfully."
SUCCESS_SETTINGS_SAVED = "Settings saved successfully."
SUCCESS_TEMPLATE_SAVED = "Template saved successfully."
SUCCESS_HISTORY_SAVED = "Entry saved to history."

# Help URLs
HELP_URL_MAIN = "https://github.com/advancedcodegenerator/docs"
HELP_URL_TEMPLATES = "https://github.com/advancedcodegenerator/docs/templates"
HELP_URL_MODELS = "https://github.com/advancedcodegenerator/docs/models"
HELP_URL_PROGRAMMING = "https://github.com/advancedcodegenerator/docs/programming"

# Import platform for OS-specific constants
import os
import platform

# OS-specific constants
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# User data directory based on OS
if IS_WINDOWS:
    USER_DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP_ORG)
elif IS_MACOS:
    USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "Library", "Application Support", APP_ORG)
else:  # Linux and others
    USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".local", "share", APP_ORG.lower())

# Ensure user data directory exists
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Default paths for data files
DEFAULT_HISTORY_PATH = os.path.join(USER_DATA_DIR, "history.json")
DEFAULT_TEMPLATES_PATH = os.path.join(USER_DATA_DIR, "templates.json")
DEFAULT_SETTINGS_PATH = os.path.join(USER_DATA_DIR, "settings.json")
DEFAULT_MODELS_CACHE_PATH = os.path.join(USER_DATA_DIR, "models")