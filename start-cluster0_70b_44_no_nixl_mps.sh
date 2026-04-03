#!/bin/bash
HOST_IP=`hostname -i`

# 定义 tmux session 的名称
SESSION_NAME="lightllm_cluster"

EXPR_NAME="70b_44_v4.1_mps"

LOGDIR="_/server_log_$EXPR_NAME"
mkdir -p $LOGDIR

# 检查该 session 是否已经存在
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  echo "正在创建新的 tmux session: $SESSION_NAME"

  # 清理残留 MPS 并重新启动
  echo quit | nvidia-cuda-mps-control 2>/dev/null
  sleep 1

  # 用所有 GPU 的 UUID 启动 MPS daemon（MPS 要求 UUID 格式）
  ALL_GPU_UUIDS=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | paste -sd,)
  export CUDA_VISIBLE_DEVICES=$ALL_GPU_UUIDS
  nvidia-cuda-mps-control -d
  echo "MPS daemon started with GPUs: $ALL_GPU_UUIDS"

  # 恢复 CUDA_VISIBLE_DEVICES，后续进程用数字索引即可（MPS daemon 已在运行）
  unset CUDA_VISIBLE_DEVICES

  # 1. 创建 session，并在后台运行。将第一个默认窗口命名为 'master'
  tmux new-session -d -s $SESSION_NAME -n master
  tmux send-keys -t $SESSION_NAME:master "unset https_proxy" C-m
  # 向 'master' 窗口发送命令并执行 (C-m 代表回车)
  tmux send-keys -t $SESSION_NAME:master "python -m lightllm.server.api_server --model_dir /mtc/wusiyu/models/llama3-70b --max_req_total_len 65536 --run_mode 'pd_master' --host $HOST_IP --port 60011 2>&1 | tee $LOGDIR/master.log" C-m

  # 2. 创建 'p0123' 窗口并发送命令
  tmux new-window -t $SESSION_NAME -n p0123
  tmux send-keys -t $SESSION_NAME:p0123 "unset https_proxy" C-m
  tmux send-keys -t $SESSION_NAME:p0123 "sleep 5; CUDA_VISIBLE_DEVICES=0,1,2,3 LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8000 --model_dir /mtc/wusiyu/models/llama3-70b --enable_mps --max_req_total_len 65536 --tp 4 --nccl_port 20010 --run_mode 'prefill' --pd_master_ip $HOST_IP --pd_master_port 60011 --host $HOST_IP 2>&1 | tee $LOGDIR/p0123.log" C-m

  # # 3. 创建 'p23' 窗口并发送命令
  # tmux new-window -t $SESSION_NAME -n p23
  # tmux send-keys -t $SESSION_NAME:p23 "sleep 40; CUDA_VISIBLE_DEVICES=2,3 LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8001 --model_dir /mtc/wusiyu/models/llama3-70b --tp 2 --nccl_port 20020 --run_mode 'prefill' --pd_master_ip $HOST_IP --pd_master_port 60011 --host $HOST_IP 2>&1 | tee $LOGDIR/p23.log" C-m

  # 5. 创建 'd4567' 窗口并发送命令
  tmux new-window -t $SESSION_NAME -n d4567
  tmux send-keys -t $SESSION_NAME:d4567 "unset https_proxy" C-m
  tmux send-keys -t $SESSION_NAME:d4567 "sleep 20; CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8003 --model_dir /mtc/wusiyu/models/llama3-70b --enable_mps --max_req_total_len 65536 --tp 4 --nccl_port 20040 --run_mode 'decode' --pd_master_ip $HOST_IP --pd_master_port 60011 --host $HOST_IP 2>&1 | tee $LOGDIR/d4567.log" C-m

  # 6. 创建 'client' 窗口并发送命令
  tmux new-window -t $SESSION_NAME -n client
  tmux send-keys -t $SESSION_NAME:client "unset https_proxy" C-m
  tmux send-keys -t $SESSION_NAME:client "cd test/benchmark/service" C-m
  tmux send-keys -t $SESSION_NAME:client "./wait_warmup.sh" C-m


  # 默认选中 master 窗口
  tmux select-window -t $SESSION_NAME:master
else
  echo "Tmux session '$SESSION_NAME' 已存在。正在直接接入..."
fi

# 接入该 session
echo "===> 使用以下命令接入 tmux session: $SESSION_NAME"
echo "tmux attach-session -t $SESSION_NAME"

if [[ "$LC_TERMINAL" == "iTerm2" ]] || [[ "$TERM_PROGRAM" == "iTerm.app" ]] || [[ -n "$ITERM_SESSION_ID" ]]; then
    echo "🍎 检测到当前终端为 iTerm2，正在使用 tmux -CC 启动原生窗口模式..."
    tmux -CC -u attach-session -t $SESSION_NAME
else
    echo "🐧 当前为常规终端环境，使用普通 tmux 模式接入..."
    tmux attach-session -t $SESSION_NAME
fi

