/**
 * 前端站点配置（前后端分离部署时，仅需修改此处，无需改动其他代码）
 *
 * 1. 同域单机部署（Nginx 反代或 FastAPI 直接托管 frontend/）：保持 API_BASE 为空字符串 ""
 * 2. 前后端分域部署（如前端放 Cloudflare Pages / Vercel，后端在云主机）：
 *    将 API_BASE 改为你的后端域名，例如 "https://api.zhijiang.example.com"（末尾不要加斜杠）
 */
window.ZHIJIANG_CONFIG = {
  API_BASE: "",
  AUTHOR_NAME: "没品人类之一",
  AUTHOR_BILIBILI_URL: "https://space.bilibili.com/396896138",
  // 创建 GitHub 仓库后填入仓库地址，例如 "https://github.com/你的用户名/zhijiang-subtitle-corpus"
  // 留空时也可自动读取后端环境变量 ZJ_GITHUB_URL
  GITHUB_REPO_URL: "",
  HOT_KEYWORDS: [
    "小作文",
    "嘉心糖",
    "贝极星",
    "奶淇琳",
    "顶碗人",
    "心球仪",
    "小海诺",
    "A-SOUL",
    "黑魔法学院",
    "猫中毒",
    "破防",
    "传说的世界"
  ]
};
