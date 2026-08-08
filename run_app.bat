@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

py -3.12 -c "import beamngpy, numpy, PyQt6" >nul 2>nul
if errorlevel 1 (
    echo Required Python packages are missing.
    echo Run install_dependencies.bat first.
    pause
    exit /b 1
)

start "BeamNG LiDAR BEV" pyw -3.12 -m beamng_lidar_bev
