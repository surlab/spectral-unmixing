
echo Now in %CD%
SET dirpath=%~dp0user_scripts
echo Moving directory to %dirpath%
cd %dirpath%
echo Now in %CD%
call conda.bat activate spectral-unmixing
call conda info --envs
call jupyter Notebook
pause