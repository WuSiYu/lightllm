#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! timeout 10 nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA driver is unavailable; refusing to start the v14 cluster." >&2
    exit 1
fi

HOST_IP="${HOST_IP:-$(hostname -i 2>/dev/null | awk '{print $1}')}"
# HOST_IP="${HOST_IP:-127.0.0.1}"
MASTER_IP="${MASTER_IP:-$HOST_IP}"
MODEL_DIR="${MODEL_DIR:-/mtc/wusiyu/models/Llama-3.3-70B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPS_PIPE="${MPS_PIPE:-/tmp/mps_prefill}"
MPS_SLOWDOWN="${MPS_SLOWDOWN:-2.0}"
SESSION_NAME="${SESSION_NAME:-lightllm_cluster_v14}"
EXPR_NAME="${EXPR_NAME:-260910-70b_p22.4d4_naive_switch-node0}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/_/server_log_$EXPR_NAME}"

MAX_REQ_TOTAL_LEN="${MAX_REQ_TOTAL_LEN:-65536}"
MAX_TOTAL_TOKEN_NUM="${MAX_TOTAL_TOKEN_NUM:-70000}"
BATCH_MAX_TOKENS="${BATCH_MAX_TOKENS:-16384}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
GRAPH_MAX_LEN_IN_BATCH="${GRAPH_MAX_LEN_IN_BATCH:-65536}"
MASTER_PORT="${MASTER_PORT:-60011}"
P01_PORT="${P01_PORT:-8000}"
P23_PORT="${P23_PORT:-8001}"
P0123_PORT="${P0123_PORT:-8002}"
DECODE_PORT="${DECODE_PORT:-8003}"
SHARED_WEIGHT_PORT_START="${SHARED_WEIGHT_PORT_START:-1300}"
FILTER="error|exception|traceback|warning|failed|oom|cuda|regist|flex|bundle|batch size|PERF"

mkdir -p "$LOGDIR" "$MPS_PIPE"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' 已存在。正在直接接入..."
    if [[ "${NO_ATTACH:-0}" == "1" ]]; then
        exit 0
    fi
    tmux attach-session -t "$SESSION_NAME"
    exit $?
fi

# Do not let stale output satisfy the client readiness check.
for log_name in master p01 p23 p0123 d4567; do
    : > "$LOGDIR/${log_name}.log"
done

# Restart the prefill-only MPS daemon on the dedicated pipe.
printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control >/dev/null 2>&1 || true
sleep 1
env CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    nvidia-cuda-mps-control -d

send_window() {
    local window="$1"
    local command="$2"
    tmux send-keys -t "$SESSION_NAME:$window" "$command" C-m
}

reset_window() {
    local window="$1"
    send_window "$window" "cd '$SCRIPT_DIR'"
    send_window "$window" "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY"
}

tmux new-session -d -s "$SESSION_NAME" -n master

# PD master running the v14 selector and its explicit policy parameters.
reset_window master
send_window master "$PYTHON_BIN -u -m lightllm.server.api_server --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --run_mode pd_master --select_p_d_node_strategy flex_tp_naive_switch --flex_tp_threshold 4000 --flex_tp_long_threshold 4000 --flex_tp_v14_long_threshold 12000 --flex_tp_slo_ttft 3 --flex_tp_mps_slowdown $MPS_SLOWDOWN --flex_tp_bundle_window_ms 20 --flex_tp_bundle_token_cap 8192 --flex_tp_bundle_token_trigger 4096 --flex_tp_max_inflight 64 --flex_tp_instance_token_credit 16384 --flex_tp_prediction_margin 0.08 --flex_tp_v14_latency_scale 1.0 --flex_tp_v14_routing_cost_weight 2.0 --flex_tp_v14_tp4_service_ratio_limit 0.60 --flex_tp_v14_tp4_pressure_threshold 1.0 --host '$HOST_IP' --port $MASTER_PORT > '$LOGDIR/master.log' 2>&1 &"
send_window master "tail -f '$LOGDIR/master.log' | grep -a --line-buffered -i -E '$FILTER'"

# TP2 master on GPUs 0,1.
tmux new-window -t "$SESSION_NAME" -n p01
reset_window p01
send_window p01 "sleep 5; env CUDA_VISIBLE_DEVICES=0,1 CUDA_MPS_PIPE_DIRECTORY='$MPS_PIPE' LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 $PYTHON_BIN -u -m lightllm.server.api_server --port $P01_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --tp 2 --max_total_token_num $MAX_TOTAL_TOKEN_NUM --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --graph_max_len_in_batch $GRAPH_MAX_LEN_IN_BATCH --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20010 --run_mode prefill --pd_master_ip '$MASTER_IP' --pd_master_port $MASTER_PORT --host '$HOST_IP' --shared_weight master --shared_weight_master_port_start $SHARED_WEIGHT_PORT_START --tp_smt_group_id flex0 --tp_smt_gpu_ids 0,1 --schedule_time_interval 0.005 > '$LOGDIR/p01.log' 2>&1 &"
send_window p01 "tail -f '$LOGDIR/p01.log' | grep -a --line-buffered -i -E '$FILTER'"

# TP2 master on GPUs 2,3; start independently so it can load in parallel.
tmux new-window -t "$SESSION_NAME" -n p23
reset_window p23
send_window p23 "sleep 6; env CUDA_VISIBLE_DEVICES=2,3 CUDA_MPS_PIPE_DIRECTORY='$MPS_PIPE' LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 $PYTHON_BIN -u -m lightllm.server.api_server --port $P23_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --tp 2 --max_total_token_num $MAX_TOTAL_TOKEN_NUM --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --graph_max_len_in_batch $GRAPH_MAX_LEN_IN_BATCH --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20020 --run_mode prefill --pd_master_ip '$MASTER_IP' --pd_master_port $MASTER_PORT --host '$HOST_IP' --shared_weight master --shared_weight_master_port_start $SHARED_WEIGHT_PORT_START --tp_smt_group_id flex0 --tp_smt_gpu_ids 2,3 --schedule_time_interval 0.005 > '$LOGDIR/p23.log' 2>&1 &"
send_window p23 "tail -f '$LOGDIR/p23.log' | grep -a --line-buffered -i -E '$FILTER'"

# TP4 slave. The delay lets TensorServer bind before TensorClient connects.
tmux new-window -t "$SESSION_NAME" -n p0123
reset_window p0123
send_window p0123 "sleep 240; env CUDA_VISIBLE_DEVICES=0,2,1,3 CUDA_MPS_PIPE_DIRECTORY='$MPS_PIPE' LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 $PYTHON_BIN -u -m lightllm.server.api_server --port $P0123_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --tp 4 --max_total_token_num $MAX_TOTAL_TOKEN_NUM --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --graph_max_len_in_batch $GRAPH_MAX_LEN_IN_BATCH --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20030 --run_mode prefill --pd_master_ip '$MASTER_IP' --pd_master_port $MASTER_PORT --host '$HOST_IP' --shared_weight slave --shared_weight_master_port_start $SHARED_WEIGHT_PORT_START --tp_smt_group_id flex0 --tp_smt_gpu_ids 0,1,2,3 --schedule_time_interval 0.005 > '$LOGDIR/p0123.log' 2>&1 &"
send_window p0123 "tail -f '$LOGDIR/p0123.log' | grep -a --line-buffered -i -E '$FILTER'"

# Decode runs on GPUs 4-7 and is outside the prefill MPS pipe.
tmux new-window -t "$SESSION_NAME" -n d4567
reset_window d4567
send_window d4567 "sleep 20; env CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 $PYTHON_BIN -u -m lightllm.server.api_server --port $DECODE_PORT --model_dir '$MODEL_DIR' --max_req_total_len $MAX_REQ_TOTAL_LEN --batch_max_tokens $BATCH_MAX_TOKENS --chunked_prefill_size $CHUNKED_PREFILL_SIZE --graph_max_len_in_batch $GRAPH_MAX_LEN_IN_BATCH --data_type bfloat16 --disable_vision --disable_audio --tp 4 --nccl_port 20040 --run_mode decode --pd_master_ip '$MASTER_IP' --pd_master_port $MASTER_PORT --host '$HOST_IP' --schedule_time_interval 0.005 --max_total_token_num 1000000 > '$LOGDIR/d4567.log' 2>&1 &"
send_window d4567 "tail -f '$LOGDIR/d4567.log' | grep -a --line-buffered -i -E '$FILTER'"

# Warmup waits for three Prefill and one Decode registration, avoiding early 417s.
tmux new-window -t "$SESSION_NAME" -n client
reset_window client
send_window client "cd '$SCRIPT_DIR/test/benchmark/service'"
send_window client "WARMUP_LONG_INPUT_TOKENS=16000 ./wait_v14_cluster_warmup.sh '$LOGDIR/master.log' 'http://$HOST_IP:60011/generate' \"\${WARMUP_TIMEOUT_S:-600}\""

tmux select-window -t "$SESSION_NAME:master"
echo "Cluster v14 started in tmux session: $SESSION_NAME"
echo "Logs: $LOGDIR"
echo "Attach with: tmux attach-session -t $SESSION_NAME"

if [[ "${NO_ATTACH:-0}" == "1" ]]; then
    exit 0
fi

if [[ "${LC_TERMINAL:-}" == "iTerm2" ]] || [[ "${TERM_PROGRAM:-}" == "iTerm.app" ]] || [[ -n "${ITERM_SESSION_ID:-}" ]]; then
    tmux -CC -u attach-session -t "$SESSION_NAME"
else
    tmux attach-session -t "$SESSION_NAME"
fi
