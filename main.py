"""
Main application entry point supporting both legacy and enhanced UIs.
Provides a unified entry point for the transcription application with all improvements.
"""

import sys
import argparse
from pathlib import Path
import os

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def run_enhanced_gui():
    """Run the enhanced GUI application with all improvements"""
    try:
        # Try to import and run the enhanced GUI
        from src.ui.enhanced_modern_gui import EnhancedTranscriptionGUI
        app = EnhancedTranscriptionGUI()
        app.run()
    except ImportError as e:
        print(f"WARNING: Enhanced GUI not available: {e}")
        print("Falling back to legacy GUI...")
        run_legacy_gui()


def run_legacy_gui():
    """Run the legacy GUI"""
    try:
        # Try to import the working legacy GUI from archive
        archive_gui = "archive/boost_and_transcribe_gui - Copy.py"
        if os.path.exists(archive_gui):
            print(f"[OK] Starting legacy GUI from {archive_gui}...")
            # Execute the working legacy GUI directly
            import subprocess
            import sys
            subprocess.run([sys.executable, archive_gui])
        elif os.path.exists("boost_and_transcribe_gui.py"):
            import boost_and_transcribe_gui
            print("[OK] Starting legacy GUI...")
            # If the legacy GUI has a main function, call it
            if hasattr(boost_and_transcribe_gui, 'main'):
                boost_and_transcribe_gui.main()
            else:
                print("WARNING: Legacy GUI found but no main function")
        else:
            print("[ERROR] Legacy GUI not found")
            print("Available files:")
            for file in os.listdir("."):
                if file.endswith(".py"):
                    print(f"   - {file}")
    except Exception as e:
        print(f"[ERROR] Could not run legacy GUI: {e}")


def check_system_info():
    """Display system information and capabilities"""
    print("\nSystem Information:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   Working directory: {os.getcwd()}")
    
    # Check for key dependencies
    dependencies = {
        "tkinter": "GUI framework",
        "pathlib": "File path handling", 
        "argparse": "Command line parsing"
    }
    
    print("\nCore Dependencies:")
    for dep, desc in dependencies.items():
        try:
            __import__(dep)
            print(f"   [OK] {dep}: {desc}")
        except ImportError:
            print(f"   [MISSING] {dep}: {desc} - NOT AVAILABLE")
    
    # Check for optional advanced dependencies
    optional_deps = {
        "PyQt6": "Enhanced GUI",
        "torch": "GPU acceleration",
        "whisper": "Transcription engine",
        "soundfile": "Audio processing",
        "ffmpeg": "Media conversion"
    }
    
    print("\nOptional Advanced Features:")
    for dep, desc in optional_deps.items():
        try:
            __import__(dep.lower() if dep != "PyQt6" else "PyQt6")
            print(f"   [OK] {dep}: {desc}")
        except ImportError:
            print(f"   [OPTIONAL] {dep}: {desc} - Install for enhanced features")


def main():
    """Main application entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Professional Audio/Video Transcription Tool v2.1")
    parser.add_argument("--gui", choices=["enhanced", "legacy"], default="enhanced",
                       help="Choose GUI version to run (default: enhanced)")
    parser.add_argument("--version", action="store_true", help="Show version and system info")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    print("Professional Audio/Video Transcription Tool v2.1.0")
    print("AI-powered transcription with GPU acceleration")
    print("=" * 60)
    
    if args.version:
        check_system_info()
        return
    
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("[DEBUG] Debug logging enabled")
    
    if args.gui == "enhanced":
        print("[ENHANCED] Starting enhanced GUI with all improvements...")
        run_enhanced_gui()
    elif args.gui == "legacy":
        print("[LEGACY] Starting legacy GUI...")
        run_legacy_gui()


if __name__ == "__main__":
    main()
