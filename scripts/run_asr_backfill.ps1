param(
    [int]$ShardIndex = 0,
    [int]$ShardTotal = 1,
    [int]$BatchSize = 14
)

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Zhijiang GPU ASR [$($ShardIndex + 1)/$ShardTotal] - RTX 5070 Ti"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  枝江直播回放 GPU 补录 - 独立进程分片 [$($ShardIndex + 1)/$ShardTotal]" -ForegroundColor Cyan
Write-Host "  独立 CUDA 上下文 + 零 GIL 争抢 | batch=$BatchSize | beam=1 | int8_float16" -ForegroundColor Green
Write-Host "  缓存与临时音频全在 E 盘 data\ 目录 | Ctrl+C 安全暂停" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

& "$Root\.venv\Scripts\python.exe" -u "$Root\run.py" backfill-asr `
    --limit 0 `
    --model large-v3-turbo `
    --compute-type int8_float16 `
    --workers 1 `
    --prefetch 1 `
    --batch-size $BatchSize `
    --beam-size 1 `
    --shard-index $ShardIndex `
    --shard-total $ShardTotal

Write-Host ""
Write-Host "[完成] 分片 [$($ShardIndex + 1)/$ShardTotal] 已全部跑完！" -ForegroundColor Green
Read-Host "按回车键退出"
