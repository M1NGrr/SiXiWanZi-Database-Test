# 四禧丸子直播语料库 — 直播回放 AI 字幕语料库与检索系统

面向二创剪辑师与社区粉丝的四禧丸子直播回放字幕语料库，覆盖 **沐霂、又一、恬豆、梨安** 4 位成员的 B 站直播回放（支持 B 站官方 AI 字幕增量抓取、本地 GPU `faster-whisper` 语音补录、专有名词纠错、同音/别称模糊检索与前后端分离部署）。

- **维护者**：`M1NG`
- **本仓库地址**：https://github.com/M1NGrr/SiXiWanZi-Database-Test
- **开源协议**：[MIT License](LICENSE)
- **云端部署参考**：见 [DEPLOY.md](DEPLOY.md)

---

## 上游开源致谢（MIT License）

本项目基于开源项目 **[RicojrH/zhijiang-database](https://github.com/RicojrH/zhijiang-database)**（枝江数据库）进行二次配置与适配开发，特此向原作者 **B 站 [`@没品人类之一`](https://space.bilibili.com/396896138)** 致谢！

- 原项目核心架构（SQLite FTS5 中文分词检索、B 站回放增量抓取、GPU Whisper 补录流水线、分级纠错与模糊检索、前后端分离与增量热同步引擎）均源自上游仓库，并完整保留原 MIT 版权声明。
- 本仓库主要完成面向 **四禧丸子（沐霂 / 又一 / 恬豆 / 梨安）** 的数据源配置、专有名词纠错与同音词表、场景识别关键词及前端视觉主题适配。

---

## 覆盖成员与数据源

| 成员 | B 站正式昵称 | UID | 粉丝名 | 应援色 | 回放列表 `series_id` |
| --- | --- | --- | --- | --- | --- |
| **沐霂** | 沐霂是MUMU呀 | `1878154667` | 小沐标 | `#A1D29A` | `2004017` |
| **又一** | 又一充电中 | `1217754423` | 酥酥又 | `#E3BA09` | `1975929` |
| **恬豆** | 恬豆发芽了 | `1660392980` | 恬豆包 | `#A7C9D3` | `1961996` |
| **梨安** | 梨安不迷路 | `1900141897` | 向心梨 | `#E38691` | `2001839` |

> **关于数据库文件**：SQLite 语料库文件体积较大，且 B 站登录 Cookie（`.env`）属于个人敏感凭证，均已加入 `.gitignore`，不包含在 Git 仓库内；请按下方步骤在本地自行抓取构建。

---

## 核心特性

- **前后端彻底分离**：
  - `frontend/`：纯静态响应式公开前端（适配手机 / 平板 / 电脑），按 4 位成员官方应援色区分角色，支持成员多选、场景多选、上下双翻页/跳页、前后 25 秒上下文展开、一键复制与直达 B 站原视频时间戳；
  - `backend/`：FastAPI 后端区分 `public_app`（公网纯只读检索接口 `/api/v1/*`，零写操作、零 Cookie 暴露）与 `local_app`（本地运维管理 `/api/v1/admin/*` 与 `/debug` 调试台）。
- **高性能中文全文检索**：SQLite WAL + FTS5 外部内容索引 + 中英混排字符级分词，支持单字、多词 AND、日期与 BV 号过滤。
- **双通道字幕采集**：
  - 直连 4 位成员 B 站「直播回放」列表（`series_id`），增量断点续抓；
  - 针对无自带字幕分 P，支持仅拉取纯音频 DASH 流 + 本地 NVIDIA GPU `faster-whisper (large-v3-turbo)` 多进程补录，转完即删临时音频、零视频囤积。
- **专有名词纠错与同音模糊搜索**：内置 [config/correction_rules.json](config/correction_rules.json) 分级纠错规则与同音/别称扩展词表（如搜 `四禧丸子` 兼搜 `四喜丸子`，搜 `小沐标` 兼搜 `小目标`，搜 `向心梨` 兼搜 `向心力`，搜 `恬豆包` 兼搜 `甜豆包`），全局繁转简（`zhconv`）。

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
│   └── config.js                # 唯一配置入口（API_BASE / 维护者 / 仓库链接 / 热门词）
├── config/
│   ├── up_list.json             # 4 位成员 UID、应援色、粉丝名与「直播回放」列表 ID
│   ├── correction_rules.json    # Tier 1/2 专有名词纠错规则与 Tier 3 同音模糊词表
│   ├── voiceprints.json         # 4 位成员 CAM++ 声纹特征库
│   └── announcements.json       # 站点公告与更新动态（实时生效）
├── sixiwanzi_corpus/            # 核心数据层（检索、抓取、GPU ASR 补录、声纹打分、增量同步）
├── scripts/                     # 本地 GPU 补录、代码部署与云端增量同步脚本
├── tests/
│   └── test_pipeline.py         # 端到端单元测试
├── DEPLOY.md                    # 云端公网部署参考教程
├── requirements.txt             # 核心与云端后端轻量依赖
├── requirements-asr.txt         # 本地 GPU Whisper 补录可选依赖
└── requirements-sv.txt          # 本地 GPU CAM++ 声纹识别可选依赖
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

# 3. 抓取直播回放 AI 字幕（默认抓取 4 人各自的「直播回放」列表，--limit 5 先试抓，0 为全量）
python run.py ingest-up all --limit 5

# 4. 可选：为无自带字幕分 P 启用本地 GPU Whisper 补录（需先安装 requirements-asr.txt）
python run.py backfill-asr --limit 0

# 5. 专有名词纠错 + 可选四成员声纹识别（需先安装 requirements-sv.txt）+ FTS5 索引优化
python run.py normalize-corpus
python run.py score-voiceprints --all-parts --limit 0
python run.py optimize-db

# 6. 启动本地服务（公开前端: http://127.0.0.1:8000/  本地运维台: http://127.0.0.1:8000/debug）
python run.py serve
```

---

## 反馈与致谢

- 欢迎在 [GitHub Issues](https://github.com/M1NGrr/SiXiWanZi-Database-Test/issues) 反馈错词、Bug 与功能建议。
- 再次感谢上游项目 **[RicojrH/zhijiang-database](https://github.com/RicojrH/zhijiang-database)** 原作者 **B 站 [`@没品人类之一`](https://space.bilibili.com/396896138)** 的开源贡献！
