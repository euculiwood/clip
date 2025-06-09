#!/bin/bash
set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. 运行训练
echo "[$(date +'%H:%M:%S')] 开始执行训练image encoder..."
cd "$SCRIPT_DIR"
python image_train.py >> /hy-tmp/log_image_encoder_2.txt 2>&1

# 2. 运行测试
echo "[$(date +'%H:%M:%S')] 开始执行测试image encoder..."
python image_test.py

# 3. 关闭系统
shutdown