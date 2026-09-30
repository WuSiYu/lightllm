#!/bin/bash
HOST_IP=$(hostname -i | awk '{print $1}')
MASTER_IP=10.120.178.75

# 定义 tmux session 的名称
SESSION_NAME="lightllm_cluster"

EXPR_NAME="260908-fixed_tp2_70b_p22d4_node1"

LOGDIR="_/server_log_$EXPR_NAME"
mkdir -p $LOGDIR
FILTER="error|exception|traceback|warning|failed|oom|cuda|regist|flex|batch size|PERF"

# 检查该 session 是否已经存在
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  echo "正在创建新的 tmux session: $SESSION_NAME"

  # 清理残留 MPS（默认 pipe 和 prefill 专用 pipe）
  echo quit | nvidia-cuda-mps-control 2>/dev/null
  CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill echo quit | nvidia-cuda-mps-control 2>/dev/null
  sleep 1

  # 仅为 prefill GPU (0-3) 启动 MPS daemon，使用独立 pipe directory
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill
  nvidia-cuda-mps-control -d
  echo "MPS daemon started for prefill GPUs (0-3) at /tmp/mps_prefill"
  unset CUDA_VISIBLE_DEVICES
  unset CUDA_MPS_PIPE_DIRECTORY

  # 1. 创建 session，并在后台运行。将第一个默认窗口命名为 'master'
  tmux new-session -d -s $SESSION_NAME -n master
  # 向 'master' 窗口发送命令并执行 (C-m 代表回车)
  # tmux send-keys -t $SESSION_NAME:master "unset http_proxy https_proxy" C-m
  # tmux send-keys -t $SESSION_NAME:master "python -m lightllm.server.api_server --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --run_mode 'pd_master' --select_p_d_node_strategy flex_tp_static_2node --flex_tp_threshold 4000 --host $HOST_IP --port 60011 > $LOGDIR/master.log 2>&1 &" C-m
  # tmux send-keys -t $SESSION_NAME:master "tail -f $LOGDIR/master.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 2. 创建 'p01' 窗口: prefill master A (tp2, GPU 0,1)
  tmux new-window -t $SESSION_NAME -n p01
  tmux send-keys -t $SESSION_NAME:p01 "unset http_proxy https_proxy" C-m
  tmux send-keys -t $SESSION_NAME:p01 "sleep 5; CUDA_VISIBLE_DEVICES=0,1 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8000 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --tp 2 --max_total_token_num 70000 --enable_mps --nccl_port 20010 --run_mode 'prefill' --pd_master_ip $MASTER_IP --pd_master_port 60011 --host $HOST_IP > $LOGDIR/p01.log 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:p01 "tail -f $LOGDIR/p01.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 3. 创建 'p23' 窗口: prefill master B (tp2, GPU 2,3)
  tmux new-window -t $SESSION_NAME -n p23
  tmux send-keys -t $SESSION_NAME:p23 "unset http_proxy https_proxy" C-m
  tmux send-keys -t $SESSION_NAME:p23 "sleep 6; CUDA_VISIBLE_DEVICES=2,3 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8001 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --tp 2 --max_total_token_num 70000 --enable_mps --nccl_port 20020 --run_mode 'prefill' --pd_master_ip $MASTER_IP --pd_master_port 60011 --host $HOST_IP > $LOGDIR/p23.log 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:p23 "tail -f $LOGDIR/p23.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  # # 4. 创建 'p0123' 窗口: prefill slave (tp4, GPU 0,2,1,3 重排对齐两组tp2)
  # tmux new-window -t $SESSION_NAME -n p0123
  # tmux send-keys -t $SESSION_NAME:p0123 "unset http_proxy https_proxy" C-m
  # tmux send-keys -t $SESSION_NAME:p0123 "sleep 8; CUDA_VISIBLE_DEVICES=0,2,1,3 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8002 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --tp 4 --max_total_token_num 70000 --enable_mps --nccl_port 20030 --run_mode 'prefill' --pd_master_ip $MASTER_IP --pd_master_port 60011 --host $HOST_IP > $LOGDIR/p0123.log 2>&1 &" C-m
  # tmux send-keys -t $SESSION_NAME:p0123 "tail -f $LOGDIR/p0123.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 5. 创建 'd4567' 窗口: decode (tp4, GPU 4,5,6,7)
  tmux new-window -t $SESSION_NAME -n d4567
  tmux send-keys -t $SESSION_NAME:d4567 "unset http_proxy https_proxy" C-m
  tmux send-keys -t $SESSION_NAME:d4567 "sleep 10; CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8003 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --tp 4 --nccl_port 20040 --run_mode 'decode' --pd_master_ip $MASTER_IP --pd_master_port 60011 --host $HOST_IP > $LOGDIR/d4567.log 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:d4567 "tail -f $LOGDIR/d4567.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  # # 6. 创建 'client' 窗口并发送命令
  # tmux new-window -t $SESSION_NAME -n client
  # tmux send-keys -t $SESSION_NAME:client "unset http_proxy https_proxy" C-m
  # tmux send-keys -t $SESSION_NAME:client "cd test/benchmark/service" C-m
  # tmux send-keys -t $SESSION_NAME:client "./wait_warmup.sh" C-m


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
