@echo off
echo Starting ThePourtrait Gallery with Email Support...
echo.

REM Set your Gmail credentials here
REM Replace YOUR_APP_PASSWORD with the 16-character app password from Gmail
set EMAIL_USER=pourtrait12@gmail.com
set EMAIL_PASS=zqas sldr cncw hyud
set TEST_MODE=false

echo Email configured for: %EMAIL_USER%
echo.

REM Get the current script directory and navigate to project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

echo Script directory: %SCRIPT_DIR%
echo Project root: %PROJECT_ROOT%

REM Navigate to the BP-work directory (where this script is located)
cd /d "%SCRIPT_DIR%"

REM Find Python executable - try multiple locations
set PYTHON_EXE=
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set PYTHON_EXE="%PROJECT_ROOT%\.venv\Scripts\python.exe"
    echo Found Python in: .venv\Scripts\python.exe
) else if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set PYTHON_EXE="%PROJECT_ROOT%\venv\Scripts\python.exe"
    echo Found Python in: venv\Scripts\python.exe
) else (
    echo Python not found! Please run setup_for_teammates.bat first
    echo.
    echo To setup:
    echo    1. Go to project root directory
    echo    2. Run: setup_for_teammates.bat
    echo    3. Then try this script again
    echo.
    pause
    exit /b 1
)

REM Start the server
echo Starting server (gallery_server1.py)...
echo.
%PYTHON_EXE% gallery_server1.py

pause