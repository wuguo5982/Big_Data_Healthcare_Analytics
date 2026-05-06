@echo off
REM Launch the FastAPI service from the project root.
uvicorn app.api:app --reload
pause
