@echo off
echo Manual FFmpeg Installation Guide
echo =================================
echo.

REM Check if FFmpeg is already installed
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo [OK] FFmpeg is already installed!
    ffmpeg -version 2>nul | findstr "ffmpeg version"
    pause
    exit /b 0
)

echo FFmpeg needs to be installed manually.
echo.
echo Please follow these steps:
echo.
echo 1. DOWNLOAD FFMPEG:
echo    Open this URL in your browser:
echo    https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
echo.
echo 2. EXTRACT THE FILE:
echo    - Extract the downloaded ZIP file
echo    - Move the extracted folder to: C:\ffmpeg
echo    - The folder should contain: C:\ffmpeg\bin\ffmpeg.exe
echo.
echo 3. ADD TO PATH:
echo    - Press Win + R, type: sysdm.cpl
echo    - Click "Environment Variables"
echo    - In "System Variables", find "Path" and click "Edit"
echo    - Click "New" and add: C:\ffmpeg\bin
echo    - Click OK on all dialogs
echo.
echo 4. RESTART:
echo    - Close this command prompt
echo    - Open a new command prompt
echo    - Test with: ffmpeg -version
echo.

set /p open_url="Would you like me to open the download URL? (y/n): "
if /i "%open_url%"=="y" (
    start https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
    echo.
    echo Download started in your browser.
    echo Follow the extraction and PATH setup instructions above.
)

echo.
echo After installation, restart your command prompt and run:
echo   python main.py
echo.
pause