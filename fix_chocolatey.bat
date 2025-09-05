@echo off
echo Fixing Chocolatey lock file issue...
echo.

REM Remove the lock file that's causing issues
if exist "C:\ProgramData\chocolatey\lib\c00565a56f0e64a50f2ea5badcb97694d43e0755" (
    echo Removing lock file...
    del /f /q "C:\ProgramData\chocolatey\lib\c00565a56f0e64a50f2ea5badcb97694d43e0755" 2>nul
    echo Lock file removed.
) else (
    echo Lock file not found - may already be resolved.
)

echo.
echo Now trying to install FFmpeg again...
choco install ffmpeg -y

echo.
echo Checking if FFmpeg is now available...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [ERROR] FFmpeg installation still failed.
    echo Please try running this script as administrator.
) else (
    echo [SUCCESS] FFmpeg is now available!
    ffmpeg -version 2>nul | findstr "ffmpeg version"
)

pause