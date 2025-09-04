#!/bin/bash

# Boost & Transcribe GUI - Installation Script
# This script installs all dependencies needed for the boost_and_transcribe_gui.py

echo "🚀 Installing Boost & Transcribe GUI dependencies..."

# Check if running on Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo "📦 Installing system dependencies (Ubuntu/Debian)..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg python3-pip python3-tk
    
elif command -v yum &> /dev/null; then
    echo "📦 Installing system dependencies (CentOS/RHEL/Fedora)..."
    sudo yum install -y ffmpeg python3-pip tkinter
    
elif command -v brew &> /dev/null; then
    echo "📦 Installing system dependencies (macOS with Homebrew)..."
    brew install ffmpeg
    
else
    echo "⚠️  Please install ffmpeg manually for your system"
    echo "   Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "   CentOS/RHEL: sudo yum install ffmpeg"
    echo "   macOS: brew install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/download.html"
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected. Python 3.8 or higher is required."
    exit 1
fi

echo "✅ Python $python_version detected"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if tkinter is available
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "❌ tkinter is not available. Please install python3-tk:"
    echo "   Ubuntu/Debian: sudo apt-get install python3-tk"
    echo "   CentOS/RHEL: sudo yum install tkinter"
    exit 1
fi

# Test if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg is not installed or not in PATH"
    exit 1
fi

echo "✅ ffmpeg is available"

# Test if whisper is available
if ! python3 -c "import whisper" 2>/dev/null; then
    echo "❌ whisper is not installed. Please run: pip3 install -r requirements.txt"
    exit 1
fi

echo "✅ whisper is available"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To run the application:"
echo "   python3 boost_and_transcribe_gui.py"
echo ""
echo "If you encounter any issues:"
echo "1. Make sure ffmpeg is in your PATH"
echo "2. Try running: pip3 install --upgrade -r requirements.txt"
echo "3. For NumPy version issues, run: pip3 install 'numpy<=2.1'" 