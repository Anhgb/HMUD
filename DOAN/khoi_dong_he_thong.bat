@echo off
chcp 65001 > nul
title Sign Language Recognition - Launcher
echo ============================================================
echo   KHOI DONG HE THONG NHAN DIEN NGON NGU KY HIEU
echo ============================================================
echo.
echo [1/2] Dang bat FastAPI Server...
start "FastAPI Server (Dong de tat AI)" cmd /c "d:\HMUD-K22\.venv\Scripts\python.exe d:\HMUD-K22\DOAN\part4_api\api_server.py & pause"

timeout /t 3 /nobreak > nul

echo [2/2] Dang bat Giao dien Desktop...
start "Sign Language App" cmd /c "d:\HMUD-K22\.venv\Scripts\python.exe d:\HMUD-K22\DOAN\part5_webapp\app_desktop.py & pause"

echo.
echo Da khoi dong xong ca 2 thanh phan!
echo Tat cua so nay neu muon.
timeout /t 3 > nul
exit
