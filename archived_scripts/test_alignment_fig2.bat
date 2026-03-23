@echo off
setlocal

REM Run from this script's directory (project root)
cd /d "%~dp0"

echo === Activating conda environment: spectral-unmixing ===
REM `conda activate` requires `call` in .bat files
call conda activate spectral-unmixing
if errorlevel 1 (
  echo.
  echo ERROR: Failed to activate conda env "spectral-unmixing".
  echo Tip: run this from an "Anaconda Prompt" or ensure conda is on PATH.
  echo.
  pause
  exit /b 1
)

echo.
echo === Testing alignment on fig2 (stacks) ===
python test_alignment_fig2.py
if errorlevel 1 (
  echo.
  echo ERROR: Alignment test failed. See messages above.
  echo.
  pause
  exit /b 1
)

echo.
echo === Test complete! ===
pause
exit /b 0





