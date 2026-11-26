# 枝江数据库 — 枝江直播回放 AI 字幕语料库与检索系统

面向二创剪辑师与社区粉丝的直播回放字幕语料库，覆盖 **嘉然、贝拉、乃琳、向晚、心宜、思诺** 6 位成员近 6 年直播回放（支持 B 站官方 AI 字幕增量抓取、本地 GPU `faster-whisper` 语音补录、专有名词纠错、同音/别称模糊检索与前后端分离云端部署）。

- **作者**：B 站 [`@没品人类之一`](https://space.bilibili.com/396896138)
- **GitHub 仓库**：https://github.com/RicojrH/zhijiang-database
- **开源协议**：[MIT License](LICENSE)
- **云端部署指南**：见 [DEPLOY.md](DEPLOY.md)
- **在线访问数据库**：https://zjdb.mprlzy.top

> **关于数据库文件**：完整的近 6 年 579 万条 SQLite 语料库约 2.12 GB，超过 GitHub 单文件限制，因此仓库仅开源完整代码、建库 Schema、纠错词表与前后端界面；你可按下方步骤自行抓取构建，或直接访问作者部署的公开检索站。

---

## 核心特性

- **前后端彻底分离**：
  - `frontend/`：纯静态响应式公开前端（适配手机 / 平板 / 电脑），按 6 位成员官方应援色区分角色，支持成员多选、场景多选、上下双翻页/跳页、前后 25 秒上下文展开、一键复制与直达 B 站原视频；
  - `backend/`：FastAPI 后端区分 `public_app`（公网纯只读检索接口 `/api/v1/*`，零写操作、零 Cookie 暴露）与 `local_app`（本地运维管理 `/api/v1/admin/*` 与 `/debug` 调试台）。
- **高性能中文全文检索**：SQLite WAL + FTS5 外部内容索引 + 中英混排字符级分词，579 万条语料实测响应 **140~190 ms**。
- **双通道字幕采集**：
  - 精准直连 6 位成员 B 站「直播回放」合集/列表（`series_id`），增量断点续抓；
  - 针对无自带字幕分 P，支持只下纯音频 DASH 流 + 本地 NVIDIA GPU `faster-whisper (large-v3-turbo)` 多进程分片补录，转完即删临时音频、零视频囤积。
- **专有名词纠错与同音模糊搜索**：内置 [config/correction_rules.json](config/correction_rules.json) 分级纠错规则与同音/别称扩展词表（如搜 `小海诺` 自动兼容 `小海螺/小螺`，搜 `心宜` 兼容未替换的 `心仪`），全局繁转简（`zhconv`）。

---

## 目录结构

```text
├── backend/                     # FastAPI 独立后端（公网只读 public_app / 本地运维 local_app）
│   ├── app.py
│   ├── debug_console.html       # 仅本地运维模式挂载在 /debug
│   └── routers/
│       ├── public.py            # 公网只读 API (/api/v1/*)
│       └── admin.py             # 本地私有运维 API (/api/v1/admin/*)
├── frontend/                    # 独立静态公开前端（可直接托管至 Cloudflare Pages / Nginx / FastAPI）
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── config.js                # 唯一配置入口（API_BASE / 作者 / GitHub 链接）
├── config/
│   ├── up_list.json             # 6 位成员 UID、应援色、粉丝名与「直播回放」合集 ID
│   └── correction_rules.json    # Tier 1/2 专有名词纠错规则与 Tier 3 同音模糊词表
├── zhijiang_corpus/             # 核心数据层（建库、抓取、清洗、GPU 补录、场景初筛、检索）
├── scripts/
│   └── run_asr_backfill.ps1     # Windows GPU 多进程分片补录脚本
├── tests/
│   └── test_pipeline.py         # 端到端单元测试
├── DEPLOY.md                    # GitHub 推送与云端公网部署完整教程
├── requirements.txt             # 核心与云端后端轻量依赖
└── requirements-asr.txt         # 本地 GPU Whisper 补录可选依赖
```

---

## 快速开始（本地开发与自建库）

```powershell
# 1. 创建虚拟环境并安装核心依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 初始化数据库 & 扫码登录 B 站（抓取 ai-zh 官方 AI 字幕需要登录态）
python run.py init-db
python run.py login

# 3. 抓取直播回放 AI 字幕（默认直接抓取 6 人各自的「直播回放」列表，--limit 0 为全量）
python run.py ingest-up all --limit 5

# 4. 可选：为无自带字幕分 P 启用本地 GPU Whisper 补录（需先安装 requirements-asr.txt）
python run.py backfill-asr --limit 0

# 5. 专有名词纠错 + 场景初筛 + FTS5 索引优化
python run.py normalize-corpus
python run.py tag-scenes --only-unknown
python run.py optimize-db

# 6. 启动本地服务（公开前端: http://127.0.0.1:8000/  本地运维台: http://127.0.0.1:8000/debug）
python run.py serve
```

---

## 致谢与反馈

欢迎通过 B 站 [`@没品人类之一`](https://space.bilibili.com/396896138) 或 GitHub Issues 反馈错词、Bug 与功能建议！
