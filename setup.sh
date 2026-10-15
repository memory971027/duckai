#!/bin/bash
# DuckAI2API 一键启动脚本（在 Claw Cloud Run 容器内执行）
set -e
cd /app

REPO="${REPO:-你的GitHub用户名/仓库名}"
BASE="https://raw.githubusercontent.com/${REPO}/main"

echo "[1/3] 安装依赖..."
pip install --no-cache-dir -r requirements.txt

echo "[2/3] 下载服务文件..."
for f in duckai.py main.py tools.py toolrouter.py; do
  curl -fsSL "${BASE}/${f}" -o "${f}"
done

echo "[3/3] 启动服务, 端口 ${PORT:-8080}..."
exec python main.py
