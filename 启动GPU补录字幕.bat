@echo off
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_asr_backfill.ps1" -ShardIndex 0 -ShardTotal 2 -BatchSize 12
timeout /t 2 /nobreak >nul
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_asr_backfill.ps1" -ShardIndex 1 -ShardTotal 2 -BatchSize 12
