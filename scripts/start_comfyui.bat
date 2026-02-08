@echo off
REM ComfyUI Startup Script for Journal App
REM This script starts ComfyUI with the API enabled for automatic image generation

echo ============================================
echo Starting ComfyUI for PrismA - Secure Journal App
echo ============================================
echo.

REM Set ComfyUI installation path
set COMFYUI_PATH=C:\ComfyUI

REM Check if ComfyUI exists
if not exist "%COMFYUI_PATH%" (
    echo ERROR: ComfyUI not found at %COMFYUI_PATH%
    echo Please install ComfyUI first: https://github.com/comfyanonymous/ComfyUI
    pause
    exit /b 1
)

REM Navigate to ComfyUI directory
cd /d "%COMFYUI_PATH%"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python and add it to your PATH
    pause
    exit /b 1
)

REM Start ComfyUI with API enabled on port 8188
echo Starting ComfyUI on http://127.0.0.1:8188 ...
echo.
echo The API will be available for automatic artwork generation.
echo Close this window to stop ComfyUI.
echo.

python main.py --listen --port 8188

REM If ComfyUI exits, pause to show any error messages
if errorlevel 1 (
    echo.
    echo ComfyUI stopped with an error.
    pause
)
