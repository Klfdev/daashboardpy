@echo off
title Dashboard – Iniciando...

echo ==============================================
echo      Iniciando o Dashboard Streamlit...
echo ==============================================
echo.

REM Caminho da pasta correta
cd /d "C:\Users\EMAM\Desktop\PY AUTUMAÇÃO"

REM Abrir o navegador automaticamente
start "" http://localhost:8501

REM Executar o Streamlit
python -m streamlit run app.py --server.headless true

exit
