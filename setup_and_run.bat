@echo off
setlocal enabledelayedexpansion

title Quantellix Order Execution Portal - Automated Setup & Launcher
echo ===============================================================================
echo     Quantellix Order Execution & AI Supply Chain Decision Platform
echo                         Automated System Setup
echo ===============================================================================
echo.

:: 1. Verify Python Installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo [INFO] Detected: %PYTHON_VER%

:: 2. Check or Create Virtual Environment (.venv)
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating a fresh .venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment (.venv) created successfully.
) else (
    echo [INFO] Existing virtual environment (.venv) found.
)

:: 3. Activate Virtual Environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 4. Install / Update Dependencies
echo [INFO] Checking and installing required packages from requirements.txt...
echo This may take a few minutes on the first run...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some dependencies had installation notices. Continuing...
)

:: 5. Ensure .env exists
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] Creating .env from .env.example...
        copy .env.example .env
        echo [NOTE] Please update your GEMINI_API_KEY in the .env file if using AI features.
    )
)

:: 6. Launch Application
echo.
echo ===============================================================================
echo     Setup Complete! Launching Quantellix Portal on Port 5000...
echo     Opening: http://127.0.0.1:5000/portal/marketplace
echo ===============================================================================
echo.

start "" "http://127.0.0.1:5000/portal/marketplace"
python -m uvicorn portal.server:app --host 127.0.0.1 --port 5000 --reload

pause
