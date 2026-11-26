param(
    [int]$Processes = 2,
    [int]$BatchSize = 8,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:HF_HOME = Join-Path $Root "data\cache\huggingface"
$env:TMP = Join-Path $Root "data\tmp_asr"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TMP | Out-Null

$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " 枝江数据库 · 旧版 ASR 高精度词级时间戳重录（支持断点续跑）" -ForegroundColor Cyan
Write-Host " 特性: word_timestamps=True | chunk_length=10s | beam=2" -ForegroundColor Gray
Write-Host " 说明: 仅重录 subtitle_lan='asr-large-v3-turbo' 的旧分P，" -ForegroundColor Gray
Write-Host "       重录完自动标为 asr-v2-large-v3-turbo，随时可 Ctrl+C 中断续跑" -ForegroundColor Gray
Write-Host "========================================================`n" -ForegroundColor Cyan

if ($Processes -le 1) {
    & $Py run.py backfill-asr --retranscribe-legacy-asr --limit $Limit --batch-size $BatchSize --beam-size 2 --chunk-length 10 --workers 1 --prefetch 1
} else {
    $jobs = @()
    for ($i = 0; $i -lt $Processes; $i++) {
        $argList = @(
            "run.py", "backfill-asr",
            "--retranscribe-legacy-asr",
            "--limit", "$Limit",
            "--batch-size", "$BatchSize",
            "--beam-size", "2",
            "--chunk-length", "10",
            "--workers", "1",
            "--prefetch", "1",
            "--shard-index", "$i",
            "--shard-total", "$Processes"
        )
        Write-Host ("启动独立 GPU 进程分片 {0}/{1}..." -f ($i + 1), $Processes) -ForegroundColor Cyan
        $jobs += Start-Process -FilePath $Py -ArgumentList $argList -WorkingDirectory $Root -NoNewWindow -PassThru
    }
    foreach ($p in $jobs) {
        $p.WaitForExit()
    }
}

Write-Host "`n[OK] 本轮高精度重录结束！运行 一键更新并同步上云.bat -SyncOnly 即可增量同步到线上。" -ForegroundColor Green
