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

REM Generate + save Figure 1.5 main figure into results/Figure1_5
python -m src.figure1_5
if errorlevel 1 (
  echo.
  echo ERROR: Figure1_5 main figure save failed. See messages above.
  echo.
  pause
  exit /b 1
)

REM Generate + save Figure 1.5 supplement panels
python -m src.figure1_5 --supplements
if errorlevel 1 (
  echo.
  echo ERROR: Figure1_5 supplement panels save failed. See messages above.
  echo.
  pause
  exit /b 1
)

echo Saved Figure 1.5 outputs (main figure and supplements) to results\Figure1_5
exit /b 0


