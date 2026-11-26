param(
    [int]$Limit = 10,
    [switch]$SkipIngest,
    [switch]$SkipAsr,
    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# 锁定所有缓存与临时目录在 E 盘项目内，绝不占用 C 盘
$env:HF_HOME = Join-Path $Root "data\cache\huggingface"
$env:TMP = Join-Path $Root "data\tmp_asr"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TMP | Out-Null

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "未找到虚拟环境 Python: $Py"
}

# 读取或首次引导配置 .cloud_sync.env（已在 .gitignore 中排除，不上传 GitHub）
$ConfFile = Join-Path $Root ".cloud_sync.env"
$Conf = @{
    CLOUD_HOST    = ""
    CLOUD_APP_DIR = "/opt/zhijiang"
    PUBLIC_URL    = "https://zjdb.mprlzy.top"
    INGEST_LIMIT  = "$Limit"
}
if (Test-Path $ConfFile) {
    Get-Content $ConfFile -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)\s*$') {
            $Conf[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
}

if (-not $Conf["CLOUD_HOST"]) {
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host " 首次运行增量同步配置（配置将保存在本地 .cloud_sync.env）" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
    $ip = (Read-Host "请输入你的云服务器公网 IP 或 root@IP（例如 47.xx.xx.xx）").Trim()
    if (-not $ip) { throw "未输入服务器地址，已取消。" }
    if ($ip -notmatch '@') { $ip = "root@$ip" }
    $Conf["CLOUD_HOST"] = $ip
    $lines = @(
        "CLOUD_HOST=$($Conf['CLOUD_HOST'])"
        "CLOUD_APP_DIR=$($Conf['CLOUD_APP_DIR'])"
        "PUBLIC_URL=$($Conf['PUBLIC_URL'])"
        "INGEST_LIMIT=$($Conf['INGEST_LIMIT'])"
    )
    [System.IO.File]::WriteAllLines($ConfFile, $lines, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "[OK] 已保存服务器配置到 $ConfFile`n" -ForegroundColor Green
}

$CloudHost   = $Conf["CLOUD_HOST"]
$CloudAppDir = $Conf["CLOUD_APP_DIR"]
$PublicUrl   = $Conf["PUBLIC_URL"].TrimEnd("/")
if ($PSBoundParameters.ContainsKey("Limit")) {
    $EffLimit = $Limit
} elseif ($Conf["INGEST_LIMIT"]) {
    $EffLimit = [int]$Conf["INGEST_LIMIT"]
} else {
    $EffLimit = 10
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " 枝江数据库 · 本地增量更新 & 极速热同步上云" -ForegroundColor Cyan
Write-Host " 目标服务器: $CloudHost ($CloudAppDir)" -ForegroundColor Gray
Write-Host " 公开站点:   $PublicUrl" -ForegroundColor Gray
Write-Host "========================================================`n" -ForegroundColor Cyan

# 同步前先抓一眼公网首页当前数据，方便结束后对比
$BeforeStats = $null
try {
    $BeforeStats = Invoke-RestMethod -Uri "$PublicUrl/api/v1/stats" -TimeoutSec 10
    Write-Host ("[同步前公网首页] 字幕: {0:N0} 条 | 有效分P: {1:N0} | 时间跨度: {2} ~ {3}" -f `
        $BeforeStats.subtitles_total, $BeforeStats.video_parts_ok, $BeforeStats.date_min, $BeforeStats.date_max) -ForegroundColor DarkGray
} catch {
    Write-Host "[提示] 暂未读取到同步前公网统计，继续执行同步..." -ForegroundColor Yellow
}

# ── 第 1 步：本地增量追新 + 语音补录 + 词表纠错 + 场景打标 ──
if (-not $SyncOnly -and -not $SkipIngest) {
    Write-Host "`n==> [1/4] 增量抓取 6 位成员最新直播回放 AI 字幕（每人检查最近 $EffLimit 场）..." -ForegroundColor Cyan
    & $Py run.py ingest-up all --limit $EffLimit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] 抓取阶段返回非零状态（可能是 B 站 Cookie 过期，可运行 python run.py login 续期）" -ForegroundColor Yellow
    }

    if (-not $SkipAsr) {
        Write-Host "`n==> [2/4] 检查无自带字幕的新分 P，调用本地 GPU faster-whisper 自动补录..." -ForegroundColor Cyan
        & $Py run.py backfill-asr --limit 0 --batch-size 12 --workers 1 --prefetch 1
    } else {
        Write-Host "`n==> [2/4] 已按参数跳过本地 GPU 语音补录 (-SkipAsr)" -ForegroundColor DarkGray
    }

    Write-Host "`n==> [3/4] 执行专有名词纠错修复..." -ForegroundColor Cyan
    & $Py run.py normalize-corpus
} else {
    Write-Host "`n==> [1-3/4] 已跳过本地抓取，直接比对并同步本地与云端的差异数据..." -ForegroundColor DarkGray
}

# ── 第 2 步：远程清单比对 + 生成增量补丁包（仅几百 KB ~ 几 MB） ──
Write-Host "`n==> [4/4] 拉取云端数据清单并计算极速增量补丁..." -ForegroundColor Cyan
$ManifestFile = Join-Path $Root "data\tmp_asr\remote_manifest.json"
$DeltaFile    = Join-Path $Root "data\tmp_asr\zhijiang_delta.json.gz"
Remove-Item -Force -ErrorAction SilentlyContinue $ManifestFile, $DeltaFile

# 先确保云端代码已包含最新增量同步命令，再拉取清单
$RemoteManifestJson = & ssh $CloudHost "cd $CloudAppDir && git pull --ff-only -q && .venv/bin/python run.py sync-manifest"
if ($LASTEXITCODE -ne 0) {
    throw "SSH 拉取云端清单失败，请检查网络连接或服务器密码/密钥。"
}
[System.IO.File]::WriteAllText($ManifestFile, ($RemoteManifestJson -join "`n"), (New-Object System.Text.UTF8Encoding($false)))

$ExportOut = & $Py run.py export-delta --remote-manifest $ManifestFile --out $DeltaFile
if ($LASTEXITCODE -ne 0) { throw "本地生成增量补丁失败。" }
$ExportInfo = ($ExportOut -join "`n") | ConvertFrom-Json

Write-Host ("    本地分P: {0} | 云端分P: {1} | 需同步分P: {2}（新增 {3}，更新 {4}）| 字幕: {5:N0} 条 | 补丁体积: {6} KB" -f `
    $ExportInfo.local_parts, $ExportInfo.remote_parts, $ExportInfo.parts_changed, `
    $ExportInfo.parts_new, $ExportInfo.parts_updated, $ExportInfo.subtitles_exported, $ExportInfo.delta_size_kb) -ForegroundColor White

if ([int]$ExportInfo.parts_changed -eq 0) {
    Remove-Item -Force -ErrorAction SilentlyContinue $ManifestFile, $DeltaFile
    Write-Host "`n[OK] 云端数据库已与本地 100% 一致，无需上传任何数据！" -ForegroundColor Green
    exit 0
}

# ── 第 3 步：秒传小补丁到香港服务器并 WAL 热写入（零停服） ──
Write-Host "`n    正在上传增量补丁 ($($ExportInfo.delta_size_kb) KB) 到 $CloudHost ..." -ForegroundColor Cyan
& scp $DeltaFile "${CloudHost}:/tmp/zhijiang_delta.json.gz"
if ($LASTEXITCODE -ne 0) { throw "SCP 上传补丁失败。" }

Write-Host "    正在云端热合并入库并增量更新 FTS5 索引（不中断公网访问）..." -ForegroundColor Cyan
$ApplyOut = & ssh $CloudHost "cd $CloudAppDir && .venv/bin/python run.py apply-delta /tmp/zhijiang_delta.json.gz && rm -f /tmp/zhijiang_delta.json.gz"
if ($LASTEXITCODE -ne 0) { throw "云端合并补丁失败。" }
$ApplyInfo = ($ApplyOut -join "`n") | ConvertFrom-Json

Remove-Item -Force -ErrorAction SilentlyContinue $ManifestFile, $DeltaFile

# ── 第 4 步：自动验收公网首页实时统计数据 ──
Write-Host "`n========================================================" -ForegroundColor Green
Write-Host " 增量热同步成功！本次写入 $($ApplyInfo.parts_applied) 个分P / $($ApplyInfo.subtitles_inserted) 条字幕" -ForegroundColor Green
try {
    $AfterStats = Invoke-RestMethod -Uri "$PublicUrl/api/v1/stats" -Headers @{ "Cache-Control" = "no-cache" } -TimeoutSec 10
    $deltaSubs = if ($BeforeStats) { $AfterStats.subtitles_total - $BeforeStats.subtitles_total } else { $ApplyInfo.subtitles_inserted }
    $deltaParts = if ($BeforeStats) { $AfterStats.video_parts_ok - $BeforeStats.video_parts_ok } else { $ApplyInfo.parts_applied }
    Write-Host (" 公网首页实时数据 ($PublicUrl)：") -ForegroundColor Green
    Write-Host ("   已收录纠错字幕 : {0:N0} 条（本次 +{1:N0}）" -f $AfterStats.subtitles_total, $deltaSubs) -ForegroundColor White
    Write-Host ("   覆盖有效分P    : {0:N0} 个分P（本次 +{1:N0}）" -f $AfterStats.video_parts_ok, $deltaParts) -ForegroundColor White
    Write-Host ("   收录时间跨度   : {0} ~ {1}" -f $AfterStats.date_min, $AfterStats.date_max) -ForegroundColor White
    Write-Host ("   库最近更新时间 : {0} (UTC)" -f $AfterStats.latest_updated_at) -ForegroundColor White
} catch {
    Write-Host " 云端库统计已更新：$($ApplyInfo.live_stats | ConvertTo-Json -Compress)" -ForegroundColor White
}
Write-Host "========================================================" -ForegroundColor Green
