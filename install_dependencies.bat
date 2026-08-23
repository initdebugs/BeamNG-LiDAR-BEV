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

rem A virtualenv PER SIMULATOR VERSION, not a global install. beamngpy 1.36
rem speaks only this simulator's bridge protocol, so the pin and
rem config.BEAMNG_EXE move together -- and a global site-packages shared with
rem another project has already broken this one twice (an unimportable PyQt6,
rem and a stray pytest-qt that loaded a second Qt runtime into pytest).
if not exist "%~dp0.venv39\Scripts\python.exe" (
    echo Creating .venv39...
    py -3.12 -m venv "%~dp0.venv39"
    if errorlevel 1 (
        echo.
        echo Could not create the virtualenv.
        pause
        exit /b 1
    )
)

echo Installing BeamNG LiDAR BEV dependencies into .venv39...
"%~dp0.venv39\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv39\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo Dependencies are installed in .venv39. Run run_app.bat to open the app.
pause
