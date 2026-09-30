#!/usr/bin/env bash
set -u

SESSION_NAME="${SESSION_NAME:-lightllm_cluster_v13}"
CLUSTER_LABEL="${CLUSTER_LABEL:-v13}"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# The repository-wide stop-cluster.sh uses a broad Python pkill. A v13
# cleanup must not terminate unrelated jobs on the same host, so only target
# the ports owned by this topology.
for pid in $(pgrep -f 'lightllm.server.api_server' 2>/dev/null || true); do
    cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' || true)
    case "$cmdline" in
        *'--port 60011 '*|*'--port 8000 '*|*'--port 8001 '*|*'--port 8002 '*|*'--port 8003 '*)
            kill -TERM "$pid" 2>/dev/null || true
            ;;
    esac
done
sleep 1
for pid in $(pgrep -f 'lightllm.server.api_server' 2>/dev/null || true); do
    cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' || true)
    case "$cmdline" in
        *'--port 60011 '*|*'--port 8000 '*|*'--port 8001 '*|*'--port 8002 '*|*'--port 8003 '*)
            kill -KILL "$pid" 2>/dev/null || true
            ;;
    esac
done

if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill \
        nvidia-cuda-mps-control >/dev/null 2>&1 || true
fi

echo "$CLUSTER_LABEL cluster cleanup complete: session=$SESSION_NAME"
