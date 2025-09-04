@echo off
REM Boost & Transcribe GUI - Windows Installation Script
REM This script installs all dependencies needed for boost_and_transcribe_gui.py

title Boost & Transcribe GUI Installer

echo Installing Boost & Transcribe GUI dependencies...

REM Ensure we're in the script's directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)
echo [OK] Python detected.

REM Ensure pip corresponds to Python 3
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available for Python.
    echo Please ensure pip is installed: python -m ensurepip
    pause
    exit /b 1
)
echo [OK] pip is available.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install Python dependencies from requirements.txt
if exist requirements.txt (
    echo Installing Python dependencies...
    python -m pip install --upgrade -r requirements.txt
) else (
    echo [ERROR] requirements.txt not found in %cd%
    pause
    exit /b 1
)

REM Check if tkinter is available
python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] tkinter is not available.
    echo Please reinstall Python with tkinter support.
    pause
    exit /b 1
)
echo [OK] tkinter is available.

echo.
echo Installation complete!
pause
