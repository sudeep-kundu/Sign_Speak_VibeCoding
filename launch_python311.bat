@echo off
setlocal enabledelayedexpansion
title DeepLabCut System - Python 3.11.9

echo.
echo ====================================================
echo    🧠 DeepLabCut System - Python 3.11.9 🧠
echo ====================================================
echo.

REM Check if Python 3.11 is available
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3.11 not found!
    echo.
    echo 📋 Please install Python 3.11.9:
    echo 1. Download from: https://www.python.org/downloads/release/python-3119/
    echo 2. Install with "Add to PATH" checked
    echo 3. Restart this script
    echo.
    echo 🌐 Opening download page...
    start https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

echo ✅ Python 3.11 found:
py -3.11 --version

echo.
echo 📦 Checking DeepLabCut dependencies...

REM Check if DeepLabCut is installed for Python 3.11
py -3.11 -c "import deeplabcut" >nul 2>&1
if errorlevel 1 (
    echo.
    echo 📥 DeepLabCut not found. Installing dependencies...
    echo This may take several minutes and requires internet connection...
    echo.
    
    py -3.11 -m pip install -r requirements_deeplabcut.txt
    
    if errorlevel 1 (
        echo.
        echo ❌ ERROR: Failed to install DeepLabCut dependencies
        echo.
        echo 💡 Troubleshooting:
        echo 1. Check your internet connection
        echo 2. Try running as administrator
        echo 3. Update pip: py -3.11 -m pip install --upgrade pip
        echo.
        pause
        exit /b 1
    )
    
    echo ✅ Dependencies installed successfully!
) else (
    echo ✅ DeepLabCut already installed
)

:main_menu
echo.
echo 🎯 DeepLabCut System Options:
echo.
echo 1. 🧠 Full DeepLabCut System (Advanced Training & Detection)
echo 2. 🎓 Simple Training Tool (Beginner-friendly)
echo 3. 🔧 Install/Update Dependencies
echo 4. 📖 View Setup Guide
echo 5. 🧪 Test DeepLabCut Installation
echo 6. ❌ Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting Full DeepLabCut System...
    echo Advanced features: Model training, pose estimation, data analysis
    echo Using Python 3.11.9 for optimal compatibility
    echo.
    py -3.11 deeplabcut_detector.py
    echo.
    echo 👋 Application closed. Returning to menu...
    pause
    goto :main_menu
) else if "%choice%"=="2" (
    echo.
    echo 🎓 Starting Simple Training Tool...
    echo Perfect for beginners: Easy data collection and training
    echo Using Python 3.11.9 for optimal compatibility
    echo.
    py -3.11 gesture_trainer_simple.py
    echo.
    echo 👋 Application closed. Returning to menu...
    pause
    goto :main_menu
) else if "%choice%"=="3" (
    echo.
    echo 🔧 Installing/Updating Dependencies...
    echo.
    py -3.11 -m pip install --upgrade pip
    py -3.11 -m pip install -r requirements_deeplabcut.txt --upgrade
    echo.
    if errorlevel 1 (
        echo ❌ Installation failed
        pause
        goto :main_menu
    ) else (
        echo ✅ Dependencies updated successfully!
        pause
        goto :main_menu
    )
) else if "%choice%"=="4" (
    echo.
    echo 📖 Opening Setup Guide...
    start DEEPLABCUT_SETUP.md
    echo Setup guide opened in your default application.
    pause
    goto :main_menu
) else if "%choice%"=="5" (
    echo.
    echo 🧪 Testing DeepLabCut Installation...
    echo.
    py -3.11 -c "import deeplabcut; print('✅ DeepLabCut version:', deeplabcut.__version__)"
    py -3.11 -c "import tensorflow as tf; print('✅ TensorFlow version:', tf.__version__)"
    py -3.11 -c "import cv2; print('✅ OpenCV version:', cv2.__version__)"
    py -3.11 -c "import numpy as np; print('✅ NumPy version:', np.__version__)"
    echo.
    echo 🎯 All core dependencies tested!
    pause
    goto :main_menu
) else if "%choice%"=="6" (
    echo.
    echo 👋 Thank you for using DeepLabCut!
    pause
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
    goto :main_menu
)

:end
echo.
echo 👋 Thanks for using the DeepLabCut System!
pause 