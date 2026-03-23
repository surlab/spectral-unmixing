@echo off
REM Activate conda environment and run Figure 2 generation

call conda activate spectral-unmixing
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'spectral-unmixing'
    exit /b 1
)

echo Generating Figure 2 subpanels...
python -m src.figure2

if errorlevel 1 (
    echo.
    echo ERROR: Figure 2 generation failed. See messages above.
    exit /b 1
)

echo.
echo Figure 2 generation complete!

