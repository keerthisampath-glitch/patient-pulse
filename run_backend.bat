@echo off
echo ====================================================================
echo      PATIENTPULSE AI — FASTAPI BACKEND SERVER LAUNCHER              
echo ====================================================================
echo Starting FastAPI server on http://127.0.0.1:8000 ...
echo Interactive Swagger Docs: http://127.0.0.1:8000/docs
echo.
python -m uvicorn App.backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
