#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$SCRIPT_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

ADDR=$(hostname -i 2>/dev/null | awk '{print $1}')
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-60011}
DURATION=${SERVEGEN_DURATION:-180}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT_S:-180}
BENCHMARK_KILL_AFTER=${BENCHMARK_KILL_AFTER_S:-20}
# LOG_DIR=${LOG_DIR:-"_/260907-70b_p22.4d4_v13_servegen"}
LOG_DIR=${LOG_DIR:-"_/260907-2N-70b_v13_servegen"}
if [[ "$LOG_DIR" != /* ]]; then
    LOG_DIR="$REPO_ROOT/$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

case "$BENCHMARK_KILL_AFTER" in
    ''|*[!0-9]*|0) echo "BENCHMARK_KILL_AFTER_S must be a positive integer" >&2; exit 2 ;;
esac

for rate in ${RATES:-20 18 16 14 12 10 8 6 4 2 1}; do
# for rate in ${RATES:-10 8 6 4 2 1}; do
    for mode in ${SERVEGEN_MODES:-mm-image m-large deepseek-r1}; do
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        out_dir="$LOG_DIR/servegen/$mode"
        mkdir -p "$out_dir"
        echo "[$(date -u)] v13 dataset=$mode rate=$rate"
        timeout --signal=TERM --kill-after="$BENCHMARK_KILL_AFTER" "${BENCHMARK_TIMEOUT_S:-900}" \
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
            python -u benchmark_serving_chat_req_rate.py \
                --dataset-type servegen \
                --servegen-mode "$mode" \
                --servegen-duration "$DURATION" \
                --servegen-output-divisor 10 \
                --request-timeout-s "$REQUEST_TIMEOUT" \
                --port "$PORT" --addr "$ADDR" --request-rate "$rate" \
                --bypass-cache --dump-dir "$out_dir" \
                2>&1 | tee "$out_dir/$rate.log"
        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
            echo "benchmark failed or timed out: dataset=$mode rate=$rate status=$status" >&2
        fi
    done
done
