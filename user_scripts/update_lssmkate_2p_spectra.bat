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

REM Update LSS-mKate1.csv with tagBFP 2P spectra
python dev_scripts/update_lssmkate_2p_spectra.py
if errorlevel 1 (
  echo.
  echo ERROR: Failed to update LSS-mKate1.csv. See messages above.
  echo.
  pause
  exit /b 1
)

echo.
echo Successfully updated LSS-mKate1.csv with mTFP1 2P excitation spectra
pause
exit /b 0

