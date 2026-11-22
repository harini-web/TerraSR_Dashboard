@echo off
cd /d "%~dp0"
set "PYRUN=python"
if exist ".venv\Scripts\python.exe" set "PYRUN=.venv\Scripts\python.exe"
echo.
echo ========= DeBlur Satellite Image Enhancer =========
echo Open your browser at http://127.0.0.1:5000
 echo.
"%PYRUN%" app.py
if errorlevel 1 (
  echo.
  echo Cannot start app. Run: python -m pip install -r requirements-dashboard.txt
  echo Python must be installed and requirements installed in your environment.
)
pause
