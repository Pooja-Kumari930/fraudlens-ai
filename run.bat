@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat ( echo Run setup.bat first. & pause & exit /b 1 )
call .venv\Scripts\activate.bat
REM Optional: put an LLM API key on the first line of llm_key.txt (never share or commit that file)
if exist llm_key.txt set /p FRAUDLENS_LLM_API_KEY=<llm_key.txt
if not exist models\fraud_pipeline.joblib ( echo Model not found - training... & python -m backend.train )
start "" http://localhost:8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
