@echo off
setlocal enabledelayedexpansion
title DeepLabCut Gesture Detection System

echo.
echo ====================================================
echo    🧠 DeepLabCut Gesture Detection System 🧠
echo ====================================================
echo.

REM Check Python version
py --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8-3.11
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found:
py --version

REM Check Python version compatibility
for /f "tokens=2" %%i in ('py --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Checking Python version compatibility...

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo ❌ Python version too old. Need Python 3.8-3.11
    echo Current version: %PYTHON_VERSION%
    pause
    exit /b 1
)

if %MAJOR% EQU 3 if %MINOR% LSS 8 (
    echo ❌ Python version too old. Need Python 3.8-3.11
    echo Current version: %PYTHON_VERSION%
    pause
    exit /b 1
)

if %MAJOR% EQU 3 if %MINOR% GTR 11 (
    :python313_menu
    echo.
    echo ⚠️  PYTHON 3.13 DETECTED!
    echo ⚠️  DeepLabCut is not compatible with Python 3.13 yet
    echo.
    echo 🔧 SOLUTIONS:
    echo 1. Install Python 3.11 alongside (recommended)
    echo 2. Use Python 3.13 compatible version (limited features)
    echo 3. Exit
    echo.
    set /p py_choice="Enter choice (1-3): "
    
    if "!py_choice!"=="1" (
        echo.
        echo 🌐 Opening Python 3.11 download page...
        start https://www.python.org/downloads/release/python-3119/
        echo.
        echo 📋 Instructions:
        echo 1. Download Python 3.11.9
        echo 2. Install with "Add to PATH" checked
        echo 3. Run this script again
        echo.
        pause
        exit /b 0
    ) else if "!py_choice!"=="2" (
        echo.
        echo 🤖 Starting Python 3.13 compatible version...
        echo Installing basic dependencies...
        
        py -m pip install -r requirements_deeplabcut_python313.txt
        
        if errorlevel 1 (
            echo ❌ Failed to install dependencies
            pause
            goto :python313_menu
        )
        
        echo ✅ Dependencies installed
        echo 🚀 Starting compatible version...
        py python313_gesture_detector.py
        echo.
        echo 👋 Application closed. Returning to menu...
        pause
        goto :python313_menu
    ) else if "!py_choice!"=="3" (
        echo Goodbye!
        pause
        exit /b 0
    ) else (
        echo Invalid choice. Please try again.
        pause
        goto :python313_menu
    )
)

echo ✅ Python version compatible

echo.
echo 📦 Checking DeepLabCut dependencies...

REM Check if DeepLabCut is installed
py -c "import deeplabcut" >nul 2>&1
if errorlevel 1 (
    echo.
    echo 📥 DeepLabCut not found. Installing dependencies...
    echo This may take several minutes and requires internet connection...
    echo.
    
    py -m pip install -r requirements_deeplabcut.txt
    
    if errorlevel 1 (
        echo.
        echo ❌ ERROR: Failed to install DeepLabCut dependencies
        echo.
        echo This usually means TensorFlow compatibility issues.
        echo.
        echo 🔧 Try these solutions:
        echo 1. Install Python 3.8-3.11
        echo 2. Use the compatible version for Python 3.13
        echo.
        echo Would you like to try the Python 3.13 compatible version?
        set /p fallback_choice="(y/n): "
        
        if /i "!fallback_choice!"=="y" (
            echo Installing Python 3.13 compatible version...
            py -m pip install -r requirements_deeplabcut_python313.txt
            
            if errorlevel 1 (
                echo ❌ Installation failed
                pause
                exit /b 1
            )
            
            echo ✅ Compatible version installed
            echo 🚀 Starting Python 3.13 compatible system...
            py python313_gesture_detector.py
            echo.
            echo 👋 Application closed. Returning to menu...
            pause
            goto :main_menu
        )
        
        pause
        exit /b 1
    )
    
    echo ✅ Dependencies installed successfully!
) else (
    echo ✅ DeepLabCut already installed
)

:main_menu
echo.
echo 🎯 Choose your option:
echo.
echo 1. 🧠 Full DeepLabCut System (Advanced)
echo 2. 🎓 Simple Training Tool (Beginner-friendly)
echo 3. 🤖 Python 3.13 Compatible Version
echo 4. 📖 View Setup Guide
echo 5. ❌ Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting Full DeepLabCut System...
    echo Advanced features: Model training, pose estimation, data analysis
    echo.
    py deeplabcut_detector.py
    echo.
    echo 👋 Application closed. Returning to menu...
    pause
    goto :main_menu
) else if "%choice%"=="2" (
    echo.
    echo 🎓 Starting Simple Training Tool...
    echo Perfect for beginners: Easy data collection and training
    echo.
    py gesture_trainer_simple.py
    echo.
    echo 👋 Application closed. Returning to menu...
    pause
    goto :main_menu
) else if "%choice%"=="3" (
    echo.
    echo 🤖 Starting Python 3.13 Compatible Version...
    echo Works with Python 3.13+, TensorFlow-based training
    echo.
    py python313_gesture_detector.py
    echo.
    echo 👋 Application closed. Returning to menu...
    pause
    goto :main_menu
) else if "%choice%"=="4" (
    echo.
    echo 📖 Opening Setup Guide...
    start DEEPLABCUT_SETUP.md
    echo Setup guide opened in your default application.
    pause
    goto :main_menu
) else if "%choice%"=="5" (
    echo Goodbye!
    pause
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
    goto :main_menu
)

:end
echo.
echo 👋 Thanks for using the DeepLabCut Gesture Detection System!
pause 