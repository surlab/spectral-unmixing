@echo off
cd /d "%~dp0"
call conda activate spectral-unmixing
python test_and_save_current.py
if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to exit.
    pause >nul
)




