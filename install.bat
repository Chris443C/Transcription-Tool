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

REM Check if FFmpeg is available
echo.
echo Checking for FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 goto ffmpeg_missing
echo [OK] FFmpeg is available.
goto ffmpeg_done

:ffmpeg_missing
echo [WARNING] FFmpeg is not found in PATH.
echo.
echo FFmpeg is REQUIRED for audio/video processing.
echo.
echo QUICK SOLUTION: Run the FFmpeg installer helper:
echo   install_ffmpeg_windows.bat
echo.
echo Or install manually:
echo Option 1 - Using Chocolatey (Recommended):
echo   1. Install Chocolatey: https://chocolatey.org/install
echo   2. Run: choco install ffmpeg
echo.
echo Option 2 - Manual Installation:
echo   1. Download FFmpeg from: https://ffmpeg.org/download.html
echo   2. Extract to a folder (e.g., C:\ffmpeg)
echo   3. Add C:\ffmpeg\bin to your PATH environment variable
echo   4. Restart this installer
echo.
set /p install_ffmpeg="Would you like to run the FFmpeg installer now? (y/n): "
if /i "%install_ffmpeg%"=="y" goto run_ffmpeg_installer
echo.
echo [INFO] The application will not work without FFmpeg.
echo You can install it later by running: install_ffmpeg_windows.bat
pause
goto ffmpeg_done

:run_ffmpeg_installer
echo.
echo Starting FFmpeg installer...
call install_ffmpeg_windows.bat
goto ffmpeg_done

:ffmpeg_done

REM Check if Whisper works
echo.
echo Testing OpenAI Whisper installation...
python -c "import whisper; print('Whisper version:', whisper.__version__)" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Whisper import failed, but this may be normal on first install.
) else (
    echo [OK] Whisper is working correctly.
)

echo.
echo =====================================
echo Installation Summary:
echo =====================================
echo [OK] Python dependencies installed
echo.
where ffmpeg >nul 2>&1
if errorlevel 1 goto summary_missing_ffmpeg
echo [OK] FFmpeg - Available
goto summary_done

:summary_missing_ffmpeg
echo [REQUIRED] FFmpeg - MISSING (see instructions above)

:summary_done
echo.
echo To run the enhanced GUI:
echo   python main.py
echo.
echo To run the legacy GUI:
echo   python main.py --gui legacy
echo.
where ffmpeg >nul 2>&1
if errorlevel 1 echo [WARNING] Please install FFmpeg before using the application.
echo =====================================
pause
