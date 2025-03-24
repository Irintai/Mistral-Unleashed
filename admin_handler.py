"""
Admin privileges handler for Advanced Code Generator
"""

import os
import sys
import ctypes
import subprocess
import tempfile

def is_admin():
    """Check if the script is running with admin privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def run_as_admin():
    """Re-run the script with admin privileges"""
    script = os.path.abspath(sys.argv[0])
    params = ' '.join([f'"{arg}"' for arg in sys.argv[1:]])
    
    try:
        if sys.executable.endswith("pythonw.exe"):
            # For GUI applications
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
        else:
            # For console applications
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
        return True
    except Exception as e:
        print(f"Error running as admin: {str(e)}")
        return False

def create_admin_shortcut():
    """Create a shortcut that runs the application as administrator"""
    try:
        # Get the path to the current script
        script_path = os.path.abspath(sys.argv[0])
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        shortcut_path = os.path.join(desktop_path, 'Advanced Code Generator (Admin).lnk')
        
        # Create temporary PowerShell script
        ps_script = f"""
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
        $Shortcut.TargetPath = "{sys.executable}"
        $Shortcut.Arguments = '"{script_path}"'
        $Shortcut.WorkingDirectory = "{os.path.dirname(script_path)}"
        $Shortcut.Save()
        
        # Set shortcut to run as administrator
        $bytes = [System.IO.File]::ReadAllBytes("{shortcut_path}")
        $bytes[0x15] = $bytes[0x15] -bor 0x20 # Set run as admin flag
        [System.IO.File]::WriteAllBytes("{shortcut_path}", $bytes)
        """
        
        # Write PowerShell script to temp file
        with tempfile.NamedTemporaryFile(suffix='.ps1', delete=False) as temp:
            temp_path = temp.name
            temp.write(ps_script.encode('utf-8'))
        
        # Execute PowerShell script
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_path], check=True)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        print(f"Admin shortcut created at: {shortcut_path}")
        return True
    except Exception as e:
        print(f"Error creating admin shortcut: {str(e)}")
        return False

def create_batch_launcher():
    """Create a batch file that launches the application as administrator"""
    try:
        # Get the path to the current script
        script_path = os.path.abspath(sys.argv[0])
        batch_path = os.path.join(os.path.dirname(script_path), 'run_as_admin.bat')
        
        # Create batch file content
        batch_content = f"""@echo off
echo Starting Advanced Code Generator with administrator privileges...
powershell -Command "Start-Process -FilePath '{sys.executable}' -ArgumentList '{script_path}' -Verb RunAs"
"""
        
        # Write batch file
        with open(batch_path, 'w') as f:
            f.write(batch_content)
        
        print(f"Admin batch launcher created at: {batch_path}")
        return True
    except Exception as e:
        print(f"Error creating batch launcher: {str(e)}")
        return False

def main():
    """Main function to ensure the application runs with admin privileges"""
    # Check if running as admin
    if is_admin():
        print("Running with administrator privileges")
        return True
    else:
        print("Not running with administrator privileges")
        
        # Ask user for admin elevation
        response = input("Would you like to restart with administrator privileges? (y/n): ")
        if response.lower() in ['y', 'yes']:
            # Try to elevate privileges
            if run_as_admin():
                # Exit the current non-elevated process
                print("Restarting with administrator privileges...")
                sys.exit(0)
            else:
                print("Failed to restart with administrator privileges")
                
                # Offer to create shortcuts
                create_shortcuts = input("Would you like to create shortcuts to run as admin? (y/n): ")
                if create_shortcuts.lower() in ['y', 'yes']:
                    create_admin_shortcut()
                    create_batch_launcher()
        
        return False

if __name__ == "__main__":
    main()