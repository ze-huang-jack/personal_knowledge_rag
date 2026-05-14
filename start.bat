@echo off
cd /d "%~dp0"
call venv\Scripts\activate
echo Starting backend (uvicorn)...
start "RAG Backend" cmd /k "cd /d %~dp0 && venv\Scripts\activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000"
echo Starting frontend (Streamlit)...
start "RAG Frontend" cmd /k "cd /d %~dp0 && venv\Scripts\activate && streamlit run streamlit_ui.py"
echo.
echo Both services started. Close the terminal windows to stop them.
echo Backend : http://localhost:8000
echo Frontend: http://localhost:8501
