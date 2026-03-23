@echo off
setlocal

REM Run from this script's directory (project root)
cd /d "%~dp0\.."

REM Activate conda env (requires CALL inside .bat files)
call conda activate spectral-unmixing
if errorlevel 1 (
  echo ERROR: Failed to activate conda env "spectral-unmixing".
  echo Tip: run this from an "Anaconda Prompt" or ensure conda is on PATH.
  echo.
  pause
  exit /b 1
)

REM Generate GCampCa-.csv file
python dev_scripts/create_gcampca_minus_csv.py
if errorlevel 1 (
  echo.
  echo ERROR: Failed to create GCampCa-.csv. See messages above.
  echo.
  pause
  exit /b 1
)

echo.
echo Successfully created GCampCa-.csv in dev_scripts/demo_data/
pause
exit /b 0



