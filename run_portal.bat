@echo off
title Quantellix Order Fulfillment Portal
echo Starting Quantellix Order Fulfillment Portal on Port 5000...

:: Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

start "" "http://127.0.0.1:5000/portal/marketplace"
python manage.py runserver 127.0.0.1:5000
pause
