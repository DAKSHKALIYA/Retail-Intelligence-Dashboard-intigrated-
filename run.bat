@echo off
REM One-click start for Windows. Double-click this file.
cd /d "%~dp0"

if not exist "Online Retail.xlsx" (
    echo ERROR: "Online Retail.xlsx" is missing in this folder.
    pause
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.11 or 3.12 and tick "Add Python to PATH".
        pause
        exit /b 1
    )
)

call venv\Scripts\activate
echo Installing libraries (first time takes a few minutes)...
pip install -r requirements.txt

if not exist models\churn_model.pkl (
    echo Training models (about 1-2 minutes)...
    python ml_models.py
)

python check_setup.py
streamlit run streamlit_app.py
pause
