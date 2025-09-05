@echo off
REM FFmpeg Installation Helper for Windows
REM This script helps install FFmpeg on Windows systems

title FFmpeg Installation Helper

echo ========================================
echo FFmpeg Installation Helper for Windows
echo ========================================
echo.

REM Check if FFmpeg is already installed
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo [OK] FFmpeg is already installed and available in PATH.
    ffmpeg -version 2>nul | findstr "ffmpeg version"
    echo.
    echo No further action needed.
    pause
    exit /b 0
)

echo FFmpeg is not found in your system PATH.
echo.
echo Choose an installation method:
echo.
echo 1. Install using Chocolatey (Recommended - Automatic)
echo 2. Manual installation (Download and setup manually)
echo 3. Exit without installing
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" goto chocolatey
if "%choice%"=="2" goto manual
if "%choice%"=="3" goto exit
goto invalid

:chocolatey
echo.
echo Installing FFmpeg using Chocolatey...
echo.

REM Check if Chocolatey is installed
where choco >nul 2>&1
if errorlevel 1 (
    echo Chocolatey is not installed. Installing Chocolatey first...
    echo.
    echo This will run a PowerShell command to install Chocolatey.
    echo Press any key to continue, or Ctrl+C to cancel.
    pause >nul
    
    REM Install Chocolatey
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    
    if errorlevel 1 (
        echo [ERROR] Failed to install Chocolatey.
        echo Please install manually or try the manual FFmpeg installation.
        pause
        exit /b 1
    )
    
    REM Refresh environment variables
    call refreshenv
    echo.
    echo Chocolatey installed successfully.
)

echo.
echo Installing FFmpeg via Chocolatey...
choco install ffmpeg -y

if errorlevel 1 (
    echo [ERROR] Failed to install FFmpeg via Chocolatey.
    echo Please try the manual installation method.
    pause
    exit /b 1
)

echo.
echo FFmpeg installation completed!
echo.
echo Please restart your command prompt or application to use FFmpeg.
goto success

:manual
echo.
echo Manual Installation Instructions:
echo =================================
echo.
echo 1. Open your web browser and go to:
echo    https://ffmpeg.org/download.html
echo.
echo 2. Click on "Windows" and choose a build (recommended: gyan.dev builds)
echo.
echo 3. Download the "release" version (not git version)
echo.
echo 4. Extract the downloaded ZIP file to a folder, for example:
echo    C:\ffmpeg\
echo.
echo 5. Add the FFmpeg bin folder to your PATH:
echo    a. Open "Edit the system environment variables"
echo    b. Click "Environment Variables"
echo    c. In "System Variables", find and select "Path", click "Edit"
echo    d. Click "New" and add: C:\ffmpeg\bin
echo    e. Click "OK" on all dialogs
echo.
echo 6. Restart your command prompt and this application
echo.
echo 7. Test by running: ffmpeg -version
echo.
echo Press any key to continue...
pause >nul
goto exit

:success
echo.
echo ========================================
echo Installation Successful!
echo ========================================
echo.
echo FFmpeg has been installed. Please:
echo 1. Restart your command prompt
echo 2. Restart the transcription application
echo 3. Run the application with: python main.py
echo.
echo To verify installation, run: ffmpeg -version
echo.
pause
exit /b 0

:invalid
echo.
echo Invalid choice. Please enter 1, 2, or 3.
echo.
goto :choice

:exit
echo.
echo Installation cancelled.
echo.
echo The transcription application requires FFmpeg to process audio/video files.
echo You can run this installer again anytime by running: install_ffmpeg_windows.bat
echo.
pause
exit /b 0