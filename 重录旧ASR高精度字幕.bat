@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\retranscribe_legacy_asr.ps1" %*
echo.
pause
