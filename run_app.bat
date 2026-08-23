@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

rem Prefer the per-simulator virtualenv. It exists because this project and
rem another once shared one global site-packages, and the collision was not
rem subtle: a PyQt6 that could not be imported at all, plus a globally
rem installed pytest-qt that dragged a second Qt runtime into the test process
rem and killed collection with 0xC0000139. The venv pins beamngpy to the one
rem version that speaks this simulator's bridge protocol, so the interpreter
rem and BEAMNG_EXE move together.
set "VENV_PY=%~dp0.venv39\Scripts\python.exe"
set "VENV_PYW=%~dp0.venv39\Scripts\pythonw.exe"

rem The quotes are PART of the value on the venv branch and absent on the
rem fallback branch, because this path contains spaces ("BeamNG.Tech Mods") and
rem the fallback is a command plus an argument. Quoting at the use site would
rem be wrong for exactly one of the two.
if exist "%VENV_PYW%" (
    set APP_PY="%VENV_PY%"
    set APP_PYW="%VENV_PYW%"
) else (
    echo No .venv39 found; falling back to the global interpreter.
    echo Run install_dependencies.bat to create it.
    set APP_PY=py -3.12
    set APP_PYW=pyw -3.12
)

%APP_PY% -c "import beamngpy, numpy, PyQt6" >nul 2>nul
if errorlevel 1 (
    echo Required Python packages are missing.
    echo Run install_dependencies.bat first.
    pause
    exit /b 1
)

start "BeamNG LiDAR BEV" %APP_PYW% -m beamng_lidar_bev
