@echo off
setlocal

REM Run from this script's directory (project root)
cd /d "%~dp0"

echo ============================================================
echo New Figure 1 + manuscript grid (debug)
echo CWD: %cd%
echo Started: %date% %time%
echo ============================================================
echo.

REM Activate conda env (requires CALL inside .bat files)
call conda activate spectral-unmixing
if errorlevel 1 (
  echo ERROR: Failed to activate conda env "spectral-unmixing".
  echo Tip: run this from an "Anaconda Prompt" or ensure conda is on PATH.
  echo.
  pause
  exit /b 1
)

echo --- Environment diagnostics (after conda activate) ---
echo CONDA_DEFAULT_ENV: %CONDA_DEFAULT_ENV%
echo CONDA_PREFIX: %CONDA_PREFIX%
echo PATH (first 250 chars): %PATH:~0,250%
echo.
echo where python:
where python
echo.
echo where pandoc:
where pandoc
echo.
echo where pdflatex:
where pdflatex
echo.
echo python --version:
python --version
echo.
echo pandoc --version:
pandoc --version
echo.
echo pdflatex --version:
pdflatex --version
echo.
echo python runtime (sys.executable / sys.prefix):
python -c "import os,sys; print('sys.executable:',sys.executable); print('sys.prefix:',sys.prefix); print('CONDA_DEFAULT_ENV:',os.environ.get('CONDA_DEFAULT_ENV'));"
echo -----------------------------------------------------
echo.

REM Generate + save both presentation_ and manuscript_ Figure 1 panels
python -m src.new_figure_1
if errorlevel 1 (
  echo.
  echo ERROR: New Figure 1 save failed. See messages above.
  echo.
  pause
  exit /b 1
)

REM Render manuscript-style grid preview (3 rows x 5 columns)
REM If MiKTeX is installed and on PATH, pdflatex is usually the most reliable choice.
python -m src.manuscript_grid --rows 3 --cols 5 --figure-number 1 --filename-prefix manuscript_ --out-basename manuscript_grid_3x5 --title "New Figure 1 manuscript grid" --pdf-engine pdflatex --panel-image-offset-cells 1
if errorlevel 1 (
  echo.
  echo ERROR: Manuscript grid render failed. See messages above.
  echo.
  pause
  exit /b 1
)

echo.
echo Saved New Figure 1 outputs to results\NewFigure1 (presentation_*, manuscript_*) and manuscript_grid_3x5.(pdf/png)
exit /b 0


