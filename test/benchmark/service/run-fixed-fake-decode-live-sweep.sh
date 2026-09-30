#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

PROFILE=${PROFILE:-fixed_tp2}
case "$PROFILE" in
    fixed_tp2|fixed_tp4|flex) ;;
    *) echo "PROFILE must be fixed_tp2, fixed_tp4, or flex" >&2; exit 2 ;;
esac
SELECTOR=${SELECTOR:-flex_tp_v6}
SESSION_NAME="lightllm_fake_decode_${PROFILE}_${SELECTOR}"
LOGDIR=${LOGDIR:-"_/260906-${PROFILE}-fake-decode-live"}
HOST_IP=${HOST_IP:-$(hostname -i 2>/dev/null | awk '{print $1}')}
HOST_IP=${HOST_IP:-127.0.0.1}
RATES=${RATES:-"10 8 6 4 2 1"}
SERVEGEN_MODES=${SERVEGEN_MODES:-"mm-image m-large deepseek-r1"}
WARMUP_TIMEOUT_S=${WARMUP_TIMEOUT_S:-600}
BENCHMARK_TIMEOUT_S=${BENCHMARK_TIMEOUT_S:-900}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-180}
BENCHMARK_KILL_AFTER_S=${BENCHMARK_KILL_AFTER_S:-20}

case "$WARMUP_TIMEOUT_S:$BENCHMARK_TIMEOUT_S:$BENCHMARK_KILL_AFTER_S" in
    *[!0-9:]*|*::*) echo "timeout settings must be positive integers" >&2; exit 2 ;;
esac
if [[ "$WARMUP_TIMEOUT_S" == 0 || "$BENCHMARK_TIMEOUT_S" == 0 || "$BENCHMARK_KILL_AFTER_S" == 0 ]]; then
    echo "timeout settings must be positive integers" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! timeout 10 nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA driver is unavailable; refusing to start fixed fake-decode sweep." >&2
    exit 1
fi
GPU_COUNT=$(nvidia-smi -L | awk 'END { print NR }')
if [[ "${GPU_COUNT:-0}" -lt 8 ]]; then
    echo "fixed live sweep requires 8 visible GPUs; found ${GPU_COUNT:-0}" >&2
    exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists; refusing to take ownership" >&2
    exit 1
fi

started=0
cleanup() {
    if [[ "$started" == 1 ]]; then
        SESSION_NAME="$SESSION_NAME" CLUSTER_LABEL="$PROFILE" \
            "$REPO_ROOT/stop-cluster-v13.sh" >/tmp/stop-fixed-live.log 2>&1 || cat /tmp/stop-fixed-live.log >&2
    fi
}
trap cleanup EXIT INT TERM

echo "starting $PROFILE fake-decode baseline: session=$SESSION_NAME GPUs=$GPU_COUNT"
if ! HOST_IP="$HOST_IP" WORKER_PROFILE="$PROFILE" SELECTOR="$SELECTOR" NO_ATTACH=1 \
    LOGDIR="$LOGDIR" bash "$REPO_ROOT/start-cluster3_70b_p22.4d4_mps_prefill_fake_decode.sh"; then
    echo "fixed fake-decode startup failed" >&2
    exit 1
fi
started=1

required_logs=("$LOGDIR/p01.log")
if [[ "$PROFILE" == "fixed_tp4" ]]; then
    required_logs=("$LOGDIR/p0123.log")
elif [[ "$PROFILE" == "flex" ]]; then
    required_logs=("$LOGDIR/p01.log" "$LOGDIR/p23.log" "$LOGDIR/p0123.log")
fi
worker_deadline=$(( $(date +%s) + WARMUP_TIMEOUT_S ))
while :; do
    all_workers_ready=1
    for worker_log in "${required_logs[@]}"; do
        if ! test -f "$worker_log" || ! grep -a -q 'server start up ok' "$worker_log"; then
            all_workers_ready=0
            break
        fi
    done
    if [[ "$all_workers_ready" == 1 ]]; then
        echo "$PROFILE all required prefill workers are ready"
        break
    fi
    if [[ "$(date +%s)" -ge "$worker_deadline" ]]; then
        echo "$PROFILE worker startup timed out; required logs: ${required_logs[*]}" >&2
        exit 1
    fi
    sleep 2
done

if ! timeout --signal=TERM --kill-after=10 "$WARMUP_TIMEOUT_S" \
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    WARMUP_LONG_INPUT_TOKENS=16000 \
    bash "$REPO_ROOT/test/benchmark/service/wait_warmup_mixed.sh" \
    --url "http://$HOST_IP:60011/generate"; then
    echo "$PROFILE mixed warmup failed or timed out" >&2
    exit 1
fi

echo "running $PROFILE ServeGen sweep: rates=$RATES modes=$SERVEGEN_MODES"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
RATES="$RATES" SERVEGEN_MODES="$SERVEGEN_MODES" LOG_DIR="$LOGDIR" \
    BENCHMARK_TIMEOUT_S="$BENCHMARK_TIMEOUT_S" REQUEST_TIMEOUT_S="$REQUEST_TIMEOUT_S" \
    BENCHMARK_KILL_AFTER_S="$BENCHMARK_KILL_AFTER_S" \
    bash "$REPO_ROOT/test/benchmark/service/loop4-sg3-v13.sh"
status=$?
if [[ "$status" -ne 0 ]]; then
    echo "$PROFILE ServeGen sweep returned status $status" >&2
    exit "$status"
fi
echo "FIXED_FAKE_DECODE_LIVE_SWEEP_OK profile=$PROFILE output=$LOGDIR"
