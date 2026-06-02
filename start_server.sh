#!/bin/bash
# Linux/macOS 一键启动脚本
DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"
python3 src/server.py
