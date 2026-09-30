#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! timeout 10 nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA driver is unavailable; refusing to start the naive cluster." >&2
  exit 1
fi

HOST_IP=${HOST_IP:-$(hostname -i 2>/dev/null | awk '{print $1}')}
HOST_IP=${HOST_IP:-127.0.0.1}
MASTER_IP=${MASTER_IP:-$HOST_IP}
MODEL_DIR=${MODEL_DIR:-/mtc/wusiyu/models/Llama-3.3-70B-Instruct}
PYTHON_BIN=${PYTHON_BIN:-python}
MPS_PIPE=${MPS_PIPE:-/tmp/mps_prefill}
MAX_REQ_TOTAL_LEN=${MAX_REQ_TOTAL_LEN:-65536}
MAX_TOTAL_TOKEN_NUM=${MAX_TOTAL_TOKEN_NUM:-70000}
BATCH_MAX_TOKENS=${BATCH_MAX_TOKENS:-16384}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
GRAPH_MAX_LEN_IN_BATCH=${GRAPH_MAX_LEN_IN_BATCH:-65536}
MASTER_PORT=${MASTER_PORT:-60011}
P01_PORT=${P01_PORT:-8000}
P23_PORT=${P23_PORT:-8001}
P0123_PORT=${P0123_PORT:-8002}
DECODE_PORT=${DECODE_PORT:-8003}
SHARED_WEIGHT_PORT_START=${SHARED_WEIGHT_PORT_START:-1200}

# 定义 tmux session 的名称
SESSION_NAME="${SESSION_NAME:-lightllm_cluster_naive}"
SELECTOR="${SELECTOR:-flex_tp_naive}"
case "$SELECTOR" in
  flex_tp_naive|flex_tp_naive_switch) ;;
  *) echo "SELECTOR must be flex_tp_naive or flex_tp_naive_switch" >&2; exit 2 ;;
esac

EXPR_NAME="70b_p22.4d4_v8_mps_prefill_only_flex_naive_4k"

LOGDIR="${LOGDIR:-_/server_log_$EXPR_NAME}"
mkdir -p "$LOGDIR"
FILTER="error|exception|traceback|warning|failed|oom|cuda|regist|flex|batch size|PERF"
TRACE_ENV=""
if [[ -n "${LIGHTLLM_MPS_TRACE_DIR:-}" ]]; then
  TRACE_ENV="LIGHTLLM_MPS_TRACE_DIR='$LIGHTLLM_MPS_TRACE_DIR' "
fi
PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS:-}"

# 检查该 session 是否已经存在
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  echo "正在创建新的 tmux session: $SESSION_NAME"

  # 清理残留 MPS（默认 pipe 和 prefill 专用 pipe）
  printf 'quit\n' | nvidia-cuda-mps-control 2>/dev/null || true
  printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill nvidia-cuda-mps-control 2>/dev/null || true
  sleep 1

  # 仅为 prefill GPU (0-3) 启动 MPS daemon，使用独立 pipe directory
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE"
  nvidia-cuda-mps-control -d
  echo "MPS daemon started for prefill GPUs (0-3) at /tmp/mps_prefill"
  unset CUDA_VISIBLE_DEVICES
  unset CUDA_MPS_PIPE_DIRECTORY

  # 1. 创建 session，并在后台运行。将第一个默认窗口命名为 'master'
  tmux new-session -d -s $SESSION_NAME -n master
  # 向 'master' 窗口发送命令并执行 (C-m 代表回车)
  tmux send-keys -t $SESSION_NAME:master "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY" C-m
  tmux send-keys -t $SESSION_NAME:master "${PYTHON_BIN} -u -m lightllm.server.api_server --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --run_mode 'pd_master' --select_p_d_node_strategy $SELECTOR --flex_tp_threshold 4000 --host $HOST_IP --port $MASTER_PORT > '$LOGDIR/master.log' 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:master "tail -f '$LOGDIR/master.log' | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 2. 创建 'p01' 窗口: prefill master A (tp2, GPU 0,1)
  tmux new-window -t $SESSION_NAME -n p01
  tmux send-keys -t $SESSION_NAME:p01 "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY" C-m
  tmux send-keys -t $SESSION_NAME:p01 "sleep 5; ${TRACE_ENV}CUDA_VISIBLE_DEVICES=0,1 CUDA_MPS_PIPE_DIRECTORY='$MPS_PIPE' LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 ${PYTHON_BIN} -u -m lightllm.server.api_server --port $P01_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --tp 2 --max_total_token_num $MAX_TOTAL_TOKEN_NUM --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --graph_max_len_in_batch $GRAPH_MAX_LEN_IN_BATCH --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20010 --run_mode 'prefill' --pd_master_ip $MASTER_IP --pd_master_port $MASTER_PORT --host $HOST_IP --shared_weight=master --shared_weight_master_port_start=$SHARED_WEIGHT_PORT_START --tp_smt_group_id flex0 --tp_smt_gpu_ids 0,1 --schedule_time_interval 0.005 ${PREFILL_EXTRA_ARGS} > '$LOGDIR/p01.log' 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:p01 "tail -f '$LOGDIR/p01.log' | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 3. 创建 'p23' 窗口: prefill master B (tp2, GPU 2,3)
  tmux new-window -t $SESSION_NAME -n p23
  tmux send-keys -t $SESSION_NAME:p23 "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY" C-m
  tmux send-keys -t $SESSION_NAME:p23 "sleep 6; ${TRACE_ENV}CUDA_VISIBLE_DEVICES=2,3 CUDA_MPS_PIPE_DIRECTORY='$MPS_PIPE' LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 ${PYTHON_BIN} -u -m lightllm.server.api_server --port $P23_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --tp 2 --max_total_token_num $MAX_TOTAL_TOKEN_NUM --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --graph_max_len_in_batch $GRAPH_MAX_LEN_IN_BATCH --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20020 --run_mode 'prefill' --pd_master_ip $MASTER_IP --pd_master_port $MASTER_PORT --host $HOST_IP --shared_weight=master --shared_weight_master_port_start=$SHARED_WEIGHT_PORT_START --tp_smt_group_id flex0 --tp_smt_gpu_ids 2,3 --schedule_time_interval 0.005 ${PREFILL_EXTRA_ARGS} > '$LOGDIR/p23.log' 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:p23 "tail -f '$LOGDIR/p23.log' | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 4. 创建 'p0123' 窗口: prefill slave (tp4, GPU 0,2,1,3 重排对齐两组tp2)
  tmux new-window -t $SESSION_NAME -n p0123
  tmux send-keys -t $SESSION_NAME:p0123 "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY" C-m
  tmux send-keys -t $SESSION_NAME:p0123 "sleep 75; ${TRACE_ENV}CUDA_VISIBLE_DEVICES=0,2,1,3 CUDA_MPS_PIPE_DIRECTORY='$MPS_PIPE' LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 ${PYTHON_BIN} -u -m lightllm.server.api_server --port $P0123_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --tp 4 --max_total_token_num $MAX_TOTAL_TOKEN_NUM --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --graph_max_len_in_batch $GRAPH_MAX_LEN_IN_BATCH --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20030 --run_mode 'prefill' --pd_master_ip $MASTER_IP --pd_master_port $MASTER_PORT --host $HOST_IP --shared_weight=slave --shared_weight_master_port_start=$SHARED_WEIGHT_PORT_START --tp_smt_group_id flex0 --tp_smt_gpu_ids 0,1,2,3 --schedule_time_interval 0.005 ${PREFILL_EXTRA_ARGS} > '$LOGDIR/p0123.log' 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:p0123 "tail -f '$LOGDIR/p0123.log' | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 5. 创建 'd4567' 窗口: decode (tp4, GPU 4,5,6,7)
  tmux new-window -t $SESSION_NAME -n d4567
  tmux send-keys -t $SESSION_NAME:d4567 "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY" C-m
  tmux send-keys -t $SESSION_NAME:d4567 "sleep 20; CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 ${PYTHON_BIN} -u -m lightllm.server.api_server --port $DECODE_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --tp 4 --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --data_type bfloat16 --disable_vision --disable_audio --nccl_port 20040 --run_mode 'decode' --pd_master_ip $MASTER_IP --pd_master_port $MASTER_PORT --host $HOST_IP --schedule_time_interval 0.005 --max_total_token_num 1000000 > '$LOGDIR/d4567.log' 2>&1 &" C-m
  tmux send-keys -t $SESSION_NAME:d4567 "tail -f '$LOGDIR/d4567.log' | grep -a --line-buffered -i -E '$FILTER'" C-m

  # 6. 创建 'client' 窗口并发送命令
  tmux new-window -t $SESSION_NAME -n client
  tmux send-keys -t $SESSION_NAME:client "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY" C-m
  tmux send-keys -t $SESSION_NAME:client "cd test/benchmark/service" C-m
  tmux send-keys -t $SESSION_NAME:client "WARMUP_LONG_INPUT_TOKENS=16000 exec timeout --signal=TERM --kill-after=10 \"${WARMUP_TIMEOUT_S:-600}\" ./wait_warmup_mixed.sh --url 'http://${HOST_IP}:${MASTER_PORT}/generate'" C-m


  # 默认选中 master 窗口
  tmux select-window -t $SESSION_NAME:master
else
  echo "Tmux session '$SESSION_NAME' 已存在。正在直接接入..."
fi

# 接入该 session
echo "===> 使用以下命令接入 tmux session: $SESSION_NAME"
echo "tmux attach-session -t $SESSION_NAME"

if [[ "${NO_ATTACH:-0}" == "1" ]]; then
  exit 0
fi

if [[ "$LC_TERMINAL" == "iTerm2" ]] || [[ "$TERM_PROGRAM" == "iTerm.app" ]] || [[ -n "$ITERM_SESSION_ID" ]]; then
    echo "🍎 检测到当前终端为 iTerm2，正在使用 tmux -CC 启动原生窗口模式..."
    tmux -CC -u attach-session -t $SESSION_NAME
else
    echo "🐧 当前为常规终端环境，使用普通 tmux 模式接入..."
    tmux attach-session -t $SESSION_NAME
fi
