#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for the live v12 sweep" >&2
    exit 1
fi
if ! timeout 10 nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA driver is unavailable; refusing to start the live v12 sweep." >&2
    exit 1
fi
GPU_COUNT=$(nvidia-smi -L | awk 'END { print NR }')
if [ "${GPU_COUNT:-0}" -lt 8 ]; then
    echo "live v12 sweep requires 8 visible GPUs; found ${GPU_COUNT:-0}" >&2
    exit 1
fi

SESSION_NAME=${SESSION_NAME:-lightllm_cluster_v12}
WARMUP_TIMEOUT_S=${WARMUP_TIMEOUT_S:-600}
LOGDIR=${LOGDIR:-"_/260906-v12-live"}
BENCH_LOG_DIR=${BENCH_LOG_DIR:-"$LOGDIR/benchmarks"}
RATES=${RATES:-"10 8 6 4 2 1"}
BENCHMARK_TIMEOUT_S=${BENCHMARK_TIMEOUT_S:-900}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-180}

case "$WARMUP_TIMEOUT_S" in
    ''|*[!0-9]*|0) echo "WARMUP_TIMEOUT_S must be positive" >&2; exit 2 ;;
esac
case "$BENCHMARK_TIMEOUT_S" in
    ''|*[!0-9]*|0) echo "BENCHMARK_TIMEOUT_S must be positive" >&2; exit 2 ;;
esac
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists; refusing to take ownership" >&2
    exit 1
fi

started=0
cleanup() {
    if [ "$started" -eq 1 ]; then
        SESSION_NAME="$SESSION_NAME" CLUSTER_LABEL=v12 \
            "$REPO_ROOT/stop-cluster-v13.sh" >/tmp/stop-v12-live.log 2>&1 || {
            cat /tmp/stop-v12-live.log >&2
        }
    fi
}
trap cleanup EXIT INT TERM

echo "starting v12 cluster: session=$SESSION_NAME GPUs=$GPU_COUNT"
if ! SESSION_NAME="$SESSION_NAME" LOGDIR="$LOGDIR" NO_ATTACH=1 \
    bash "$REPO_ROOT/start-cluster4_70b_p22.4d4_mps_flex_v12.sh"; then
    echo "v12 cluster startup failed" >&2
    exit 1
fi
started=1

deadline=$((SECONDS + WARMUP_TIMEOUT_S))
while [ "$SECONDS" -lt "$deadline" ]; do
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "v12 tmux session exited before warmup completed" >&2
        exit 1
    fi
    pane=$(tmux capture-pane -p -t "$SESSION_NAME:client" -S -80 2>/dev/null || true)
    if printf '%s\n' "$pane" | grep -q 'mixed warmup complete'; then
        echo "v12 mixed warmup complete"
        break
    fi
    if printf '%s\n' "$pane" | grep -q 'mixed warmup timed out\|Timed out waiting'; then
        echo "$pane" >&2
        exit 1
    fi
    sleep 2
done
if [ "$SECONDS" -ge "$deadline" ]; then
    echo "v12 warmup exceeded ${WARMUP_TIMEOUT_S}s" >&2
    exit 1
fi

echo "running v12 ServeGen sweep: rates=$RATES"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
RATES="$RATES" LOG_DIR="$BENCH_LOG_DIR" \
    BENCHMARK_TIMEOUT_S="$BENCHMARK_TIMEOUT_S" REQUEST_TIMEOUT_S="$REQUEST_TIMEOUT_S" \
    bash "$REPO_ROOT/test/benchmark/service/loop4-sg3.sh"
status=$?
if [ "$status" -ne 0 ]; then
    echo "v12 ServeGen sweep returned status $status" >&2
    exit "$status"
fi
echo "V12_LIVE_SWEEP_OK output=$LOGDIR"
