#!/usr/bin/env bash
# 枝江数据库 — Ubuntu 22.04/24.04 云端公网只读服务一键部署脚本
# 用法（在服务器上以 root 运行）：
#   bash /opt/zhijiang/scripts/deploy_cloud.sh
# 或首次直接：
#   curl -fsSL https://raw.githubusercontent.com/RicojrH/zhijiang-database/main/scripts/deploy_cloud.sh | sudo bash

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/RicojrH/zhijiang-database.git}"
APP_DIR="${APP_DIR:-/opt/zhijiang}"
WORKERS="${WORKERS:-2}"

if [ "$(id -u)" -ne 0 ]; then
  echo "请使用 root 或 sudo 运行本脚本"
  exit 1
fi

echo "==> [1/7] 检查并配置 2G Swap（防止小内存机器 OOM）"
if ! swapon --show | grep -q '/swapfile'; then
  if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> [2/7] 安装基础依赖 (git / python3-venv / nginx)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3 python3-venv python3-pip curl nginx

echo "==> [3/7] 拉取或更新仓库到 ${APP_DIR}"
if [ -d "${APP_DIR}/.git" ]; then
  git -C "${APP_DIR}" pull --ff-only
else
  mkdir -p "${APP_DIR}"
  git clone "${REPO_URL}" "${APP_DIR}"
fi
mkdir -p "${APP_DIR}/data"

echo "==> [4/7] 安装 Python 轻量运行环境"
if [ ! -d "${APP_DIR}/.venv" ]; then
  python3 -m venv "${APP_DIR}/.venv"
fi
"${APP_DIR}/.venv/bin/pip" install --upgrade pip -q
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt" -q

echo "==> [5/7] 检查并解压数据库文件"
if [ -f "${APP_DIR}/data/zhijiang_subtitles.db.tar.gz" ]; then
  echo "    发现压缩包，正在解压到 ${APP_DIR}/data/zhijiang_subtitles.db ..."
  tar -xzf "${APP_DIR}/data/zhijiang_subtitles.db.tar.gz" -C "${APP_DIR}/data"
  rm -f "${APP_DIR}/data/zhijiang_subtitles.db.tar.gz"
  echo "    解压完成并已清理压缩包。"
elif [ -f "${APP_DIR}/data/zhijiang_subtitles.db" ]; then
  echo "    已检测到现有数据库 ${APP_DIR}/data/zhijiang_subtitles.db"
else
  echo "    [提示] 尚未上传数据库文件，服务将先以空库启动；上传解压后执行 systemctl restart zhijiang-api 即可生效。"
fi

echo "==> [6/7] 配置并启动 systemd 只读后端服务 (zhijiang-api)"
cat > /etc/systemd/system/zhijiang-api.service <<EOF
[Unit]
Description=Zhijiang Subtitle Corpus Public Read-Only API
After=network.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
Environment="ZJ_ENABLE_ADMIN=0"
Environment="ZJ_SERVE_FRONTEND=1"
Environment="ZJ_DB_PATH=${APP_DIR}/data/zhijiang_subtitles.db"
Environment="ZJ_CORS_ORIGINS=*"
Environment="ZJ_GITHUB_URL=https://github.com/RicojrH/zhijiang-database"
ExecStart=${APP_DIR}/.venv/bin/uvicorn backend.app:public_app --host 127.0.0.1 --port 8000 --workers ${WORKERS}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable zhijiang-api
systemctl restart zhijiang-api

echo "==> [7/7] 配置 Nginx 80 端口反向代理（开启 gzip 压缩提速）"
cat > /etc/nginx/sites-available/zhijiang <<'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    gzip on;
    gzip_min_length 512;
    gzip_comp_level 5;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30s;
    }
}
EOF

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/zhijiang /etc/nginx/sites-enabled/zhijiang
nginx -t
systemctl enable nginx
systemctl restart nginx

sleep 2
echo ""
echo "=========================================="
echo " 健康检查结果："
curl -fsS http://127.0.0.1/api/v1/health && echo ""
echo " 数据库统计："
curl -fsS http://127.0.0.1/api/v1/stats | head -c 320 && echo "..."
PUB_IP="$(curl -fsS --max-time 3 https://api.ipify.org 2>/dev/null || echo '<你的公网IP>')"
echo ""
echo " 部署完成！现在直接在浏览器访问："
echo "   http://${PUB_IP}/"
echo "=========================================="
