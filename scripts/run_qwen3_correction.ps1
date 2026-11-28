<#
.SYNOPSIS
  四禧丸子语料库 Qwen3-ASR-1.7B 高性能分片回溯校正一键流水线
  （支持双进程 GPU 分片并行 -> GPU 声纹重打分 -> 新词表全库纠错 -> FTS5 索引优化）
.NOTES
  - 支持随时优雅暂停：新建 data\tmp_asr\PAUSE_QWEN3 文件，所有分片在当前分 P 落库后即安全停止；
    删掉该文件后重跑本脚本自动断点续跑。
  - 实时 UTF-8 日志写入 data\tmp_asr\qwen3_p1_shard*.log 与 data\tmp_asr\qwen3_correction.log
#>
param(
    [ValidateSet("p0", "p1", "both", "quality-flagged")]
    [string]$Mode = "p1",
    [int]$BatchSize = 24,
    [double]$MaxBatchSec = 180,
    [int]$MaxAlignTokens = 2400,
    [int]$Shards = 2,
    [int]$Prefetch = 3
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root "data\tmp_asr"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$MainLog = Join-Path $LogDir "qwen3_correction.log"
$PauseFile = Join-Path $LogDir "PAUSE_QWEN3"

function Write-Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Output $line
    [System.IO.File]::AppendAllText($MainLog, $line + "`n", [System.Text.UTF8Encoding]::new($false))
}

$QwenPy = Join-Path $Root ".venv-qwen-asr\Scripts\python.exe"
$MainPy = Join-Path $Root ".venv\Scripts\python.exe"

function Invoke-QwenSharded([string]$RunMode) {
    if ($Shards -le 1) {
        & $QwenPy scripts\qwen3_backfill.py --mode $RunMode --batch-size $BatchSize --max-batch-sec $MaxBatchSec --max-align-tokens $MaxAlignTokens --prefetch $Prefetch --log-file $MainLog
        return
    }
    Write-Log "启动 $Shards 路 GPU 并行分片 (mode=$RunMode, batch=$BatchSize, max_batch_sec=$MaxBatchSec, align_tok=$MaxAlignTokens, prefetch=$Prefetch)..."
    $procs = @()
    for ($s = 0; $s -lt $Shards; $s++) {
        $sLog = Join-Path $LogDir ("qwen3_${RunMode}_shard${s}.log")
        $argList = @(
            "scripts\qwen3_backfill.py",
            "--mode", $RunMode,
            "--batch-size", "$BatchSize",
            "--max-batch-sec", "$MaxBatchSec",
            "--max-align-tokens", "$MaxAlignTokens",
            "--prefetch", "$Prefetch",
            "--shard-index", "$s",
            "--shard-total", "$Shards",
            "--log-file", $sLog
        )
        $p = Start-Process -FilePath $QwenPy -ArgumentList $argList -WorkingDirectory $Root -NoNewWindow -PassThru
        $procs += $p
        Write-Log "  分片 S${s}/${Shards} 已启动 (PID=$($p.Id), log=$sLog)"
        if ($s -lt ($Shards - 1)) { Start-Sleep -Seconds 5 }
    }
    foreach ($p in $procs) {
        $p.WaitForExit()
        Write-Log "  分片进程 PID=$($p.Id) 已退出 (ExitCode=$($p.ExitCode))"
    }
}

if ($Mode -eq "p0" -or $Mode -eq "both") {
    Write-Log "=== 阶段 1: P0 异常/残缺分 P Qwen3-ASR-1.7B 校正 ==="
    Invoke-QwenSharded "p0"
}

if (-not (Test-Path $PauseFile) -and ($Mode -eq "p1" -or $Mode -eq "both")) {
    Write-Log "=== 阶段 2: P1 全部剩余 Whisper 分 P Qwen3-ASR-1.7B 高速分片升级 ==="
    Invoke-QwenSharded "p1"
}

if (-not (Test-Path $PauseFile) -and $Mode -eq "quality-flagged") {
    Write-Log "=== 质检预警/无字幕分 P Qwen3-ASR-1.7B 补录 ==="
    Invoke-QwenSharded "quality-flagged"
}

if (Test-Path $PauseFile) {
    Write-Log "检测到 PAUSE_QWEN3，已优雅暂停，跳过后续声纹与索引优化。"
    exit 0
}

Write-Log "=== 收口 1/3: 为全部新校正分 P 执行 GPU 声纹重打分 ==="
$svOut = & $MainPy run.py score-voiceprints --limit 0 2>&1 | Out-String
[System.IO.File]::AppendAllText($MainLog, $svOut + "`n", [System.Text.UTF8Encoding]::new($false))

Write-Log "=== 收口 2/3: 执行全库专有名词纠错 (normalize-corpus，跳过人工锁定行) ==="
$normOut = & $MainPy run.py normalize-corpus 2>&1 | Out-String
[System.IO.File]::AppendAllText($MainLog, $normOut + "`n", [System.Text.UTF8Encoding]::new($false))

Write-Log "=== 收口 3/3: 压缩优化 FTS5 索引与回收 WAL ==="
$optOut = & $MainPy run.py optimize-db 2>&1 | Out-String
[System.IO.File]::AppendAllText($MainLog, $optOut + "`n", [System.Text.UTF8Encoding]::new($false))

Write-Log "=== 全部 Qwen3-ASR 高速校正与收口流水线已完成 ==="
