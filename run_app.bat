@echo off
REM One-click launcher for the Silver Prediction Streamlit app.
REM Place this file directly inside D:\SILVER PREDICTION\main (same folder as venv\)

cd /d "%~dp0"

call venv\Scripts\activate.bat

REM Launch Streamlit in its own window so you can see logs / close it to stop the server
start "Silver Prediction Server" cmd /k streamlit run ui\app.py

REM Give the server a few seconds to boot before opening the browser
timeout /t 5 /nobreak >nul

start "" http://localhost:8501
