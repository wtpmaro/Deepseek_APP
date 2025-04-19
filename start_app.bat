@echo off
title DeepSeek Data Analyzer
echo Uruchamianie aplikacji DeepSeek Data Analyzer...
echo.
echo Proszę czekać, trwa inicjalizacja...
echo.
echo Po uruchomieniu, aplikacja otworzy się automatycznie w przeglądarce.
echo Nie zamykaj tego okna podczas korzystania z aplikacji!
echo.

:: Sprawdzenie, czy Python jest zainstalowany
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python nie został znaleziony. Proszę zainstalować Python i spróbować ponownie.
    echo.
    pause
    exit /b 1
)

:: Sprawdzenie, czy Streamlit jest zainstalowany
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalowanie wymaganych bibliotek...
    pip install streamlit pandas matplotlib requests
)

:: Uruchomienie aplikacji Streamlit
echo Uruchamianie aplikacji...
start "" http://localhost:8501
streamlit run app.py

pause