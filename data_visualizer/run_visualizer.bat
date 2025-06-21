@echo off
echo Launching SEU Detector Data Visualizer...
echo.

REM Navigate to the script directory
cd /d "%~dp0"

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in your PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import dash, pandas, plotly" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing required packages...
    pip install dash pandas plotly
    if %ERRORLEVEL% neq 0 (
        echo Error installing packages
        pause
        exit /b 1
    )
)

echo Starting dashboard...
python app.py
pause
