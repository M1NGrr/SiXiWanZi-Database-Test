/**
 * 前端站点配置（前后端分离部署时，仅需修改此处，无需改动其他代码）
 *
 * 1. 同域单机部署（Nginx / Caddy 反代或 FastAPI 直接托管 frontend/）：保持 API_BASE 为空字符串 ""
 * 2. 前后端分域部署（如前端放 Cloudflare Pages / Vercel，后端在云主机）：
 *    将 API_BASE 改为你的后端域名，例如 "https://api.example.com"（末尾不要加斜杠）
 */
window.SIXIWANZI_CONFIG = {
  API_BASE: "",
  AUTHOR_NAME: "M1NG",
  // 若后续有个人主页链接可填写于此，留空时展示纯文本署名
  AUTHOR_HOME_URL: "https://github.com/M1NGrr",
  GITHUB_REPO_URL: "https://github.com/M1NGrr/SiXiWanZi-Database-Test",
  // 上游开源项目与原作者署名（MIT License）
  UPSTREAM_AUTHOR_NAME: "没品人类之一",
  UPSTREAM_AUTHOR_URL: "https://space.bilibili.com/396896138",
  UPSTREAM_REPO_URL: "https://github.com/RicojrH/zhijiang-database",
  HOT_KEYWORDS: [
    "黑话",
    "爆",
    "羞死了",
    "四禧丸子",
    "小沐标",
    "酥酥又",
    "恬豆包",
    "向心梨",
    "大沐博士",
    "唐豆人"
  ]
};
