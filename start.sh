#!/usr/bin/env bash

# FAW 一键点火护卫舰脚本
# 仅用于拉起网页控制台服务器

cd "$(dirname "$0")"

echo "======================================"
echo "    启动 FAW 可视化中控操作台"
echo "======================================"

PYTHON_EXEC="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "❌ 致命错误: 未安装指定的 Python3 ($PYTHON_EXEC)"
    exit 1
fi

echo "🚀 引擎全开！开启 Uvicorn 异步服务网关..."
echo "访问 http://127.0.0.1:8000 即可接管总机。"
echo "======================================"

$PYTHON_EXEC web_app.py > /dev/null 2>&1 &
echo "🚀 后台服务已拉起，日志已重定向！"
echo "停止服务请执行: pkill -f web_app.py"
