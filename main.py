"""
Main application entry point supporting both legacy and enhanced UIs.
Provides a unified entry point for the transcription application with all improvements.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.enhanced_modern_gui import EnhancedTranscriptionGUI
from src.legacy.compatibility_layer import monkey_patch_legacy_gui


def run_enhanced_gui():
    """Run the enhanced GUI application with all improvements"""
    app = EnhancedTranscriptionGUI()
    app.run()


def run_legacy_gui():
    """Run the legacy GUI with new architecture integration"""
    # Apply monkey patch for new architecture integration
    if monkey_patch_legacy_gui():
        print("✅ Legacy GUI enhanced with new architecture")
    else:
        print("⚠️  Running legacy GUI without enhancements")
    
    # Import and run legacy GUI
    try:
        from boost_and_transcribe_gui import BoostTranscribeGUI
        import tkinter as tk
        root = tk.Tk()
        app = BoostTranscribeGUI(root)
        root.mainloop()
    except ImportError as e:
        print(f"❌ Could not load legacy GUI: {e}")
        print("🔄 Falling back to enhanced GUI...")
        run_enhanced_gui()


def main():
    """Main application entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Audio/Video Transcription Tool v2.1")
    parser.add_argument("--gui", choices=["enhanced", "legacy"], default="enhanced",
                       help="Choose GUI version to run (default: enhanced)")
    parser.add_argument("--version", action="version", version="%(prog)s 2.1.0")
    
    args = parser.parse_args()
    
    print("🎤 Audio/Video Transcription Tool v2.1.0")
    print("Enhanced with parallel processing, error handling, and modern UI")
    print("=" * 65)
    
    if args.gui == "enhanced":
        print("🚀 Starting enhanced GUI with all improvements...")
        run_enhanced_gui()
    elif args.gui == "legacy":
        print("🔄 Starting legacy GUI with modern architecture...")
        run_legacy_gui()


if __name__ == "__main__":
    main()
