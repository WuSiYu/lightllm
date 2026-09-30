#!/bin/bash
HOST_IP=$(hostname -i | awk '{print $1}')
FLEX_TP_VERSION="${FLEX_TP_VERSION:-v3}"
FLEX_TP_MPS_SLOWDOWN="${FLEX_TP_MPS_SLOWDOWN:-2.0}"
SESSION_NAME="lightllm_cluster_${FLEX_TP_VERSION}"
EXPR_NAME="70b_p22.4d4_v8_mps_prefill_flex_${FLEX_TP_VERSION}"
LOGDIR="_/server_log_$EXPR_NAME"
mkdir -p "$LOGDIR"
FILTER="error|exception|traceback|warning|failed|oom|cuda|regist|flex|bundle|batch size|PERF"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Tmux session '$SESSION_NAME' 已存在。正在直接接入..."
else
  echo "正在创建新的 tmux session: $SESSION_NAME"
  echo quit | nvidia-cuda-mps-control 2>/dev/null
  echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill nvidia-cuda-mps-control 2>/dev/null
  sleep 1

  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill
  nvidia-cuda-mps-control -d
  unset CUDA_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY

  tmux new-session -d -s "$SESSION_NAME" -n master
  tmux send-keys -t "$SESSION_NAME:master" "unset http_proxy https_proxy" C-m
  tmux send-keys -t "$SESSION_NAME:master" "python -m lightllm.server.api_server --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --run_mode pd_master --select_p_d_node_strategy flex_tp_${FLEX_TP_VERSION} --flex_tp_slo_ttft 3 --flex_tp_long_threshold 4000 --flex_tp_mps_slowdown $FLEX_TP_MPS_SLOWDOWN --flex_tp_bundle_window_ms 20 --flex_tp_bundle_token_cap 8192 --flex_tp_bundle_token_trigger 4096 --flex_tp_max_inflight 64 --flex_tp_instance_token_credit 16384 --flex_tp_prediction_margin 0.08 --host $HOST_IP --port 60011 > $LOGDIR/master.log 2>&1 &" C-m
  tmux send-keys -t "$SESSION_NAME:master" "tail -f $LOGDIR/master.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  tmux new-window -t "$SESSION_NAME" -n p01
  tmux send-keys -t "$SESSION_NAME:p01" "unset http_proxy https_proxy" C-m
  tmux send-keys -t "$SESSION_NAME:p01" "sleep 5; CUDA_VISIBLE_DEVICES=0,1 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8000 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --tp 2 --max_total_token_num 70000 --batch_max_tokens 16384 --chunked_prefill_size 8192 --graph_max_len_in_batch 65536 --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20010 --run_mode prefill --pd_master_ip $HOST_IP --pd_master_port 60011 --host $HOST_IP --shared_weight master --shared_weight_master_port_start 1300 --tp_smt_group_id flex0 --tp_smt_gpu_ids 0,1 --schedule_time_interval 0.005 > $LOGDIR/p01.log 2>&1 &" C-m
  tmux send-keys -t "$SESSION_NAME:p01" "tail -f $LOGDIR/p01.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  tmux new-window -t "$SESSION_NAME" -n p23
  tmux send-keys -t "$SESSION_NAME:p23" "unset http_proxy https_proxy" C-m
  tmux send-keys -t "$SESSION_NAME:p23" "until grep -a -q 'server start up ok' $LOGDIR/p01.log; do sleep 2; done; CUDA_VISIBLE_DEVICES=2,3 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8001 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --tp 2 --max_total_token_num 70000 --batch_max_tokens 16384 --chunked_prefill_size 8192 --graph_max_len_in_batch 65536 --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20020 --run_mode prefill --pd_master_ip $HOST_IP --pd_master_port 60011 --host $HOST_IP --shared_weight master --shared_weight_master_port_start 1300 --tp_smt_group_id flex0 --tp_smt_gpu_ids 2,3 --schedule_time_interval 0.005 > $LOGDIR/p23.log 2>&1 &" C-m
  tmux send-keys -t "$SESSION_NAME:p23" "tail -f $LOGDIR/p23.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  tmux new-window -t "$SESSION_NAME" -n p0123
  tmux send-keys -t "$SESSION_NAME:p0123" "unset http_proxy https_proxy" C-m
  tmux send-keys -t "$SESSION_NAME:p0123" "until grep -a -q 'server start up ok' $LOGDIR/p01.log && grep -a -q 'server start up ok' $LOGDIR/p23.log; do sleep 2; done; CUDA_VISIBLE_DEVICES=0,2,1,3 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8002 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --tp 4 --max_total_token_num 70000 --batch_max_tokens 16384 --chunked_prefill_size 8192 --graph_max_len_in_batch 65536 --data_type bfloat16 --disable_vision --disable_audio --enable_mps --nccl_port 20030 --run_mode prefill --pd_master_ip $HOST_IP --pd_master_port 60011 --host $HOST_IP --shared_weight slave --shared_weight_master_port_start 1300 --tp_smt_group_id flex0 --tp_smt_gpu_ids 0,1,2,3 --schedule_time_interval 0.005 > $LOGDIR/p0123.log 2>&1 &" C-m
  tmux send-keys -t "$SESSION_NAME:p0123" "tail -f $LOGDIR/p0123.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  tmux new-window -t "$SESSION_NAME" -n d4567
  tmux send-keys -t "$SESSION_NAME:d4567" "unset http_proxy https_proxy" C-m
  tmux send-keys -t "$SESSION_NAME:d4567" "sleep 20; CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=12 LIGHTLLM_TOKEN_MAX_BYTES=16384 python -m lightllm.server.api_server --port 8003 --model_dir /mtc/wusiyu/models/Llama-3.3-70B-Instruct --max_req_total_len 65536 --batch_max_tokens 16384 --chunked_prefill_size 8192 --graph_max_len_in_batch 65536 --data_type bfloat16 --disable_vision --disable_audio --tp 4 --nccl_port 20040 --run_mode decode --pd_master_ip $HOST_IP --pd_master_port 60011 --host $HOST_IP --schedule_time_interval 0.005 > $LOGDIR/d4567.log 2>&1 &" C-m
  tmux send-keys -t "$SESSION_NAME:d4567" "tail -f $LOGDIR/d4567.log | grep -a --line-buffered -i -E '$FILTER'" C-m

  tmux new-window -t "$SESSION_NAME" -n client
  tmux send-keys -t "$SESSION_NAME:client" "unset http_proxy https_proxy" C-m
  tmux send-keys -t "$SESSION_NAME:client" "cd test/benchmark/service" C-m
  tmux send-keys -t "$SESSION_NAME:client" "./wait_warmup.sh" C-m
  tmux select-window -t "$SESSION_NAME:master"
fi

echo "tmux attach-session -t $SESSION_NAME"
tmux attach-session -t "$SESSION_NAME"
