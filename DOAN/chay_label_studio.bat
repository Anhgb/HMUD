@echo off
chcp 65001 > nul
title Label Studio - Sign Language Project
echo.
echo ============================================================
echo   KHOI DONG LABEL STUDIO
echo   Truy cap: http://localhost:8080
echo   De dung: Nhan Ctrl+C trong cua so nay
echo ============================================================
echo.

set LABEL_STUDIO_BASE_DATA_DIR=D:\HMUD-K22\DOAN\label_studio_data
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:\HMUD-K22\DOAN\part1_thu_thap_du_lieu\raw_videos

d:\HMUD-K22\DOAN\.venv_labelstudio\Scripts\label-studio.exe start --port 8080

pause
