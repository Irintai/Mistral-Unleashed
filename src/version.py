"""
Version information for the Advanced Code Generator.
"""

# Application version
VERSION = "2.0.0"

# Version components
VERSION_MAJOR = 2
VERSION_MINOR = 0
VERSION_PATCH = 0

# Release type: alpha, beta, rc, or final
RELEASE_TYPE = "final"

# Build number, incremented for each build
BUILD_NUMBER = 1

# Full version display string
VERSION_DISPLAY = f"v{VERSION}"
if RELEASE_TYPE != "final":
    VERSION_DISPLAY += f"-{RELEASE_TYPE}"

# Check if this is a development version
IS_DEV_VERSION = RELEASE_TYPE != "final"

def get_version_info():
    """Get complete version information as a dictionary"""
    return {
        "version": VERSION,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "release_type": RELEASE_TYPE,
        "build": BUILD_NUMBER,
        "display": VERSION_DISPLAY,
        "is_dev": IS_DEV_VERSION
    }