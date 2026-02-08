@echo off
REM Setup ComfyUI Auto-Start on Windows Boot
REM This script adds ComfyUI to Windows Task Scheduler to start automatically

echo ============================================
echo ComfyUI Auto-Start Setup
echo ============================================
echo.
echo This will configure ComfyUI to start automatically when Windows boots.
echo.

REM Check for admin privileges
net session >nul 2>&1
if errorlevel 1 (
    echo ERROR: Administrator privileges required!
    echo Please right-click and "Run as administrator"
    pause
    exit /b 1
)

REM Set paths
set COMFYUI_PATH=C:\ComfyUI
set SCRIPT_PATH=%~dp0start_comfyui.bat
set TASK_NAME=ComfyUI-AutoStart

REM Check if ComfyUI exists
if not exist "%COMFYUI_PATH%" (
    echo ERROR: ComfyUI not found at %COMFYUI_PATH%
    echo Please install ComfyUI first and verify the path in this script.
    pause
    exit /b 1
)

REM Remove existing task if it exists
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo Removing existing task...
    schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
)

REM Create new scheduled task
echo Creating scheduled task...
schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT_PATH%\"" /sc onlogon /rl highest /f

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create scheduled task!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo ComfyUI will now start automatically when you log in.
echo.
echo To manually start ComfyUI now, run:
echo   %SCRIPT_PATH%
echo.
echo To disable auto-start, open Task Scheduler and delete:
echo   Task: %TASK_NAME%
echo.
pause
