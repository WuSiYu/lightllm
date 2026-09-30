#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

ADDR=$(hostname -i 2>/dev/null | awk '{print $1}')
ADDR=${ADDR:-127.0.0.1}
PORT=60011
SLEEP_INTERVAL=10
# LOG_DIR="_/260409-70b_p4d4_v8_mps_simple.1-5"
LOG_DIR="_/260910-70b_naive_switch_simple.1-5"
# LOG_DIR="_/260908-2N-70b_v14_simple.1-5"
# LOG_DIR="_/260908-2N-70b_fixed_tp4_simple.1-5"
REQUEST_TIMEOUT=${REQUEST_TIMEOUT_S:-180}
BENCHMARK_TIMEOUT=${BENCHMARK_TIMEOUT_S:-900}
BENCHMARK_KILL_AFTER=${BENCHMARK_KILL_AFTER_S:-20}

case "$BENCHMARK_KILL_AFTER" in
    ''|*[!0-9]*|0) echo "BENCHMARK_KILL_AFTER_S must be a positive integer" >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"

# for rate in 36 34 32 30 28 26 24 22 20 16 14 12 10 8 6; do
for rate in 18 17 16 15 14 13 12 11 10 8 5; do
    echo "============================================"
    echo "[$(date)] Starting benchmark with request-rate=$rate"
    echo "============================================"

    for dataset in simple.1-5; do
    # for dataset in simple.1-0 simple.1-3 simple.1-5 simple.1-10 simple.1-20; do
        echo "--------------------------------------------"
        echo "[$(date)] Running benchmark with dataset=$dataset"
        echo "--------------------------------------------"

        mkdir -p "${LOG_DIR}/${dataset}"
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        set +e
        timeout --signal=TERM --kill-after="$BENCHMARK_KILL_AFTER" "$BENCHMARK_TIMEOUT" \
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
            python -u benchmark_serving_chat_req_rate.py \
            --num-prompts 1000 \
            --dataset-type="$dataset" \
            --request-timeout-s "$REQUEST_TIMEOUT" \
            --port "$PORT" \
            --addr "$ADDR" \
            --request-rate "$rate" \
            --bypass-cache \
            --dump-dir "$LOG_DIR/${dataset}" \
            2>&1 | tee "${LOG_DIR}/${dataset}/${rate}.log"
        benchmark_status=${PIPESTATUS[0]}
        set -e
        if [ "$benchmark_status" -ne 0 ]; then
            echo "[$(date)] benchmark failed or timed out (status=$benchmark_status); continuing" >&2
        fi

    done

    echo "[$(date)] Finished rate=$rate"

    if [ "$rate" -lt 20 ]; then
        echo "Sleeping ${SLEEP_INTERVAL}s before next run..."
        sleep "$SLEEP_INTERVAL"
    fi
done

echo "============================================"
echo "[$(date)] All benchmarks completed!"
echo "============================================"
