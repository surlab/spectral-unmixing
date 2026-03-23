@echo off
setlocal

REM Run from this script's directory (project root)
cd /d "%~dp0"

REM Activate conda env (requires CALL inside .bat files)
call conda activate spectral-unmixing
if errorlevel 1 (
  echo ERROR: Failed to activate conda env "spectral-unmixing".
  echo Tip: run this from an "Anaconda Prompt" or ensure conda is on PATH.
  echo.
  pause
  exit /b 1
)

REM Generate + save all Figure 5 subpanels into results/Figure5
python -m src.figure5
if errorlevel 1 (
  echo.
  echo ERROR: Figure5 save failed. See messages above.
  echo.
  pause
  exit /b 1
)

echo Saved Figure 5 outputs to results\Figure5
exit /b 0

