#!/bin/bash

# 定义 tmux session 的名称
SESSION_NAME="lightllm_cluster"

echo "⏳ 正在停止 tmux session: $SESSION_NAME ..."
tmux kill-session -t $SESSION_NAME 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Tmux session '$SESSION_NAME' 已成功关闭。"
else
    echo "⚠️ 找不到名为 '$SESSION_NAME' 的 tmux session，它可能已经被关闭。"
fi

echo "🧹 正在清理可能残留的 LightLLM 进程，以释放显存和端口..."
# 使用 pkill 根据命令行关键字精准匹配并击杀相关进程

pkill -f "/opt/conda/bin/python"

if [ $? -eq 0 ]; then
    echo "✅ 发现并清理了残留的 LightLLM 进程。"
    sleep 1
    pkill -f "/opt/conda/bin/python"

    if [ $? -eq 0 ]; then
        echo "依然存活"
        sleep 1
        pkill -9 -f "/opt/conda/bin/python"
        if [ $? -eq 0 ]; then
            echo "✅ 强制清理了残留的 LightLLM 进程。"
        else
            echo "没有发现需要强制清理的残留进程。"
        fi
    fi
else
    echo "没有发现需要清理的残留进程。"
fi

echo "🧹 正在清理可能残留的 gunicorn 进程，以释放显存和端口..."
# 使用 pkill 根据命令行关键字精准匹配并击杀相关进程
pkill -f "gunicorn"

if [ $? -eq 0 ]; then
    echo "✅ 发现并清理了残留的 gunicorn 进程。"
else
    echo "没有发现需要清理的残留进程。"
fi

echo "🎉 集群清理工作全部完成！"

sleep 0.3

netstat -lntp
