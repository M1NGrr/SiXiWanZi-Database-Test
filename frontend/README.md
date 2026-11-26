# Frontend（独立前端预留目录）

本目录预留给后续面向公网用户的独立前端站点（如 Vue 3 / React / Next.js / SvelteKit 等静态部署至 Cloudflare Pages / Vercel / Nginx）。

## 对接后端只读 API 契约（`backend/routers/public.py`）

云端部署后端时，使用只读实例启动：

```bash
uvicorn backend.app:public_app --host 0.0.0.0 --port 8000
```

前端仅需请求以下 **纯只读 JSON 接口**（完整 OpenAPI Schema 见后端 `/docs` 与 `/openapi.json`）：

| 方法 | 路径 | 用途 |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | 健康检查 |
| `GET` | `/api/v1/members` | 获取 4 位成员列表与 `scene_types` 枚举（渲染前端筛选器） |
| `GET` | `/api/v1/stats` | 展示首页语料规模统计（总场次、总字幕条数、成员分布） |
| `GET` | `/api/v1/search` | 核心全文检索（`q`, `up_name`, `date_from`, `date_to`, `scene_type`, `bvid`, `context_lines`, `sort`, `limit`, `offset`） |
| `GET` | `/api/v1/videos` | 分页浏览已入库的直播回放列表 |
| `GET` | `/api/v1/videos/{bvid}/subtitles` | 按时间窗口读取某场直播的前后上下文字幕 |

> 说明：`/api/v1/admin/*` 与 `/debug` 仅在本地运维模式开启，云端 `public_app` 下完全不存在，保证公网只读安全。
