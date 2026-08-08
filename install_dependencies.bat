@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher "py" was not found.
    echo Install Python 3.12, then run this file again.
    pause
    exit /b 1
)

echo Installing BeamNG LiDAR BEV dependencies...
py -3.12 -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo Dependencies are installed. Run run_app.bat to open the app.
pause
