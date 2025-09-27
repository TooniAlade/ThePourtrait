@echo off
echo 🔧 ThePourtrait Gallery - Team Setup Script
echo ==========================================
echo.

REM Get directories
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

echo 📁 Setting up in: %PROJECT_ROOT%
echo.

REM Navigate to project root
cd /d "%PROJECT_ROOT%"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.8+ from: https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if virtual environment exists
if exist ".venv" (
    echo ✅ Virtual environment already exists
) else (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment and install requirements
echo 🔧 Installing dependencies...
call .venv\Scripts\activate.bat
if exist "BP-work\requirements.txt" (
    pip install -r BP-work\requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
) else (
    echo ❌ requirements.txt not found in BP-work folder
    pause
    exit /b 1
)

echo.
echo 🎉 Setup complete!
echo.
echo 🚀 To run the gallery:
echo    cd BP-work
echo    start_with_email.bat
echo.
echo 🌐 Then open: http://localhost:5000
echo.

pause