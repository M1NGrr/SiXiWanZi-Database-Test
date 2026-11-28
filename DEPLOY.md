# 云端部署与 GitHub 开源指南

本指南帮助你完成两件事：
1. **将本项目源码开源到 GitHub**（不含 2.12 GB 数据库文件与私人 Cookie）；
2. **将 579 万条字幕库 + 前后端安全部署到云端**，供所有人通过手机、平板、电脑直接访问。

---

## 一、开源到 GitHub（不含数据库与 Cookie）

> `sixiwanzi_subtitles.db` 体积超过 GitHub 单文件 100 MB 硬限制，**不放进 Git 仓库**（`.gitignore` 已自动排除 `.env`、`data/*.db*`、`data/cache/`、`data/tmp_asr/`、`data/models/`、`.venv/`）。GitHub 只开源代码、建库 Schema、纠错词表与前后端。

1. 在浏览器打开 https://github.com/new ，新建一个公开仓库（例如命名为 `SiXiWanZi-Database-Test`），**不要**勾选初始化 README / LICENSE（本地已准备好）。
2. 在项目根目录运行：

```powershell
git remote add origin https://github.com/M1NGrr/SiXiWanZi-Database-Test.git
git branch -M main
git push -u origin main
```

3. 推完后，把你的仓库地址填进以下任意一处，前端「关于与作者」页就会自动显示可点击的 GitHub 链接：
   - 前端：[frontend/config.js](frontend/config.js) 中的 `GITHUB_REPO_URL`
   - 或云端环境变量：`ZJ_GITHUB_URL=https://github.com/M1NGrr/SiXiWanZi-Database-Test`

---

## 二、云端服务器选型建议（性价比最高）

针对你当前 **2.12 GB SQLite + FTS5 索引（579 万行，只读检索）**：

- **推荐配置**：**2 核 CPU / 4 GB 内存 / 40 GB SSD 硬盘**（系统 + SQLite 索引常驻页面缓存后，检索可稳定在 50~180ms）
- **参考价格**：
  - 国内云（腾讯云 / 阿里云轻量应用服务器）：约 ¥35~60/月（需域名备案才能绑定 80/443 端口；若走境外如香港/新加坡轻量，无需备案，国内直连延迟低）
  - 海外 VPS（RackNerd / Hetzner / Vultr 等）：约 $3~6/月
- **不需要 GPU！** 云端只跑纯只读检索后端（`backend.app:public_app`），平时追新与 GPU 补录继续在你本地 5070 Ti 电脑跑，定期增量同步 `.db` 文件上云即可。

---

## 三、推荐部署架构（两种方式任选）

### 方案 A（最省心 · 单台云主机一体部署，推荐入门）

一台 2C4G Linux（Ubuntu 22.04/24.04）主机，用 **Caddy**（自动申请免费 HTTPS 证书）反代 **FastAPI 只读后端**（`/` 自动托管 `frontend/` 公开前端，`/api/v1/*` 提供只读 API，**不暴露任何 `/api/v1/admin/*` 与 `/debug`**）：

#### 1. 上传代码与数据库到服务器

```bash
# 服务器上拉取你刚推送的 GitHub 仓库
git clone https://github.com/M1NGrr/SiXiWanZi-Database-Test.git /opt/sixiwanzi
cd /opt/sixiwanzi
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

在本地 Windows PowerShell 把 `sixiwanzi_subtitles.db` 压缩上传（压成 `.gz` 后体积大幅缩小，上传更快）：

```powershell
# 本地打包上传（需先将服务器 IP 替换为你的真实 IP）
tar -czf data/sixiwanzi_subtitles.db.tar.gz -C data sixiwanzi_subtitles.db
scp data/sixiwanzi_subtitles.db.tar.gz root@<服务器IP>:/opt/sixiwanzi/data/
```

服务器端解压：

```bash
cd /opt/sixiwanzi/data
tar -xzf sixiwanzi_subtitles.db.tar.gz
rm -f sixiwanzi_subtitles.db.tar.gz
```

#### 2. 配置 systemd 常驻只读后端服务

创建 `/etc/systemd/system/sixiwanzi-api.service`：

```ini
[Unit]
Description=SiXiWanZi Subtitle Corpus Public Read-Only API
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/sixiwanzi
Environment="ZJ_ENABLE_ADMIN=0"
Environment="ZJ_SERVE_FRONTEND=1"
Environment="ZJ_DB_PATH=/opt/sixiwanzi/data/sixiwanzi_subtitles.db"
Environment="ZJ_CORS_ORIGINS=https://你的域名.com"
Environment="ZJ_GITHUB_URL=https://github.com/M1NGrr/SiXiWanZi-Database-Test"
ExecStart=/opt/sixiwanzi/.venv/bin/uvicorn backend.app:public_app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启动并设为开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sixiwanzi-api
sudo systemctl status sixiwanzi-api
```

#### 3. 用 Caddy 一键配置域名与自动 HTTPS

安装 Caddy 后编辑 `/etc/caddy/Caddyfile`：

```caddy
你的域名.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8000
}
```

重载即可获得自动 HTTPS：

```bash
sudo systemctl reload caddy
```

---

### 方案 B（高并发进阶 · 前端托管 Cloudflare Pages + 后端独立云主机）

如果你希望全球访问更快、前端零带宽费用：
1. 把 `frontend/config.js` 里的 `API_BASE` 改为你的后端域名，例如 `"https://api.你的域名.com"`；
2. 登录 [Cloudflare Pages](https://pages.cloudflare.com/) 连接你的 GitHub 仓库，将 **Build output directory** 设为 `frontend`（无需构建命令，纯静态零成本部署！）；
3. 云主机只负责 `api.你的域名.com`（设 `ZJ_SERVE_FRONTEND=0`、`ZJ_CORS_ORIGINS=https://你的前端域名.com`），并可在 Cloudflare 开启基础限流防刷。

---

## 四、后续定期更新云端数据（5 分钟增量同步）

每过 1~2 周在本地电脑跑一次追新：

```powershell
.\.venv\Scripts\python.exe run.py ingest-up all --limit 10
.\.venv\Scripts\python.exe run.py backfill-asr --limit 0
.\.venv\Scripts\python.exe run.py normalize-corpus
.\.venv\Scripts\python.exe run.py score-voiceprints --all-parts --limit 0
.\.venv\Scripts\python.exe run.py optimize-db
```

然后用 `rsync`（仅传增量块，通常几十秒）或 `scp` 替换云端 `/opt/sixiwanzi/data/sixiwanzi_subtitles.db` 并执行 `sudo systemctl restart sixiwanzi-api` 即可无缝生效。
