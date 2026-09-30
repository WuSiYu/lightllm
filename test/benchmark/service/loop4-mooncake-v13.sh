#!/usr/bin/env bash
set -u

CALLER_DIR=$(pwd)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

DATASET=${MOONCAKE_DATASET:?set MOONCAKE_DATASET to a Mooncake JSONL trace}
case "$DATASET" in
    /*) ;;
    *) DATASET="$CALLER_DIR/$DATASET" ;;
esac
ADDR=$(hostname -i 2>/dev/null | awk '{print $1}')
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-60011}
DURATION=${MOONCAKE_DURATION:-180}
NUM_PROMPTS=${MOONCAKE_NUM_PROMPTS:-1000}
OUTPUT_DIVISOR=${MOONCAKE_OUTPUT_DIVISOR:-10}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT_S:-180}
BENCHMARK_TIMEOUT=${BENCHMARK_TIMEOUT_S:-900}
BENCHMARK_KILL_AFTER=${BENCHMARK_KILL_AFTER_S:-20}
LOG_DIR=${LOG_DIR:-"_/260905-70b_p22.4d4_v13_mooncake"}

case "$BENCHMARK_KILL_AFTER" in
    ''|*[!0-9]*|0) echo "BENCHMARK_KILL_AFTER_S must be a positive integer" >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"

for rate in ${RATES:-10 8 6 4 2 1}; do
    out_dir="$LOG_DIR/rate-$rate"
    mkdir -p "$out_dir"
    echo "[$(date -u)] v13 mooncake rate=$rate dataset=$DATASET"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    set +e
    timeout --signal=TERM --kill-after="$BENCHMARK_KILL_AFTER" "$BENCHMARK_TIMEOUT" \
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
        python -u benchmark_serving_chat_req_rate.py \
            --dataset-type mooncake \
            --dataset "$DATASET" \
            --num-prompts "$NUM_PROMPTS" \
            --mooncake-output-divisor "$OUTPUT_DIVISOR" \
            --request-timeout-s "$REQUEST_TIMEOUT" \
            --port "$PORT" --addr "$ADDR" --request-rate "$rate" \
            --bypass-cache --dump-dir "$out_dir" \
            2>&1 | tee "$out_dir/benchmark.log"
    benchmark_status=${PIPESTATUS[0]}
    set -e
    if [ "$benchmark_status" -ne 0 ]; then
        echo "[$(date -u)] benchmark failed or timed out (status=$benchmark_status); continuing" >&2
    fi
done
