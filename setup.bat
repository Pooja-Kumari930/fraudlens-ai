@echo off
REM One-time setup on Windows: creates a virtual environment, installs packages, trains the models.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10+ not found. Install it from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist .venv ( python -m venv .venv || (echo Could not create venv & pause & exit /b 1) )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install fastapi "uvicorn[standard]" pandas numpy scikit-learn joblib || (echo Package install failed & pause & exit /b 1)
echo.
echo Training models (about 1-3 minutes)...
python -m backend.train || (echo Training failed & pause & exit /b 1)
python -m backend.source_audit || (echo Source audit failed & pause & exit /b 1)
echo.
echo Setup finished. Now double-click run.bat
pause
