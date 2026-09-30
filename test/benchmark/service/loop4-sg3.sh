#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

ADDR=$(hostname -i 2>/dev/null | awk '{print $1}')
ADDR=${ADDR:-127.0.0.1}
PORT=60011
SLEEP_INTERVAL=10
# LOG_DIR="_/260409-70b_p4d4_v8_mps_sg2"
# LOG_DIR="_/260409-70b_p22.4d4_v8_mps_flex_naive_4k_sg2"
# LOG_DIR="${LOG_DIR:-_/260907-70b_p22d4_fixed_tp2_sg2}"
# LOG_DIR="${LOG_DIR:-_/260907-70b_p22.4d4_v13_sg2}"
# LOG_DIR="${LOG_DIR:-_/260908-2N-70b_v14_sg2}"
# LOG_DIR="${LOG_DIR:-_/260908-2N-70b_fixed_tp4_sg2}"
LOG_DIR="${LOG_DIR:-_/260910-70b_naive_switch_sg2}"
# LOG_DIR="${LOG_DIR:-_/260908-2N-70b_v13_sg2}"
REQUEST_TIMEOUT=${REQUEST_TIMEOUT_S:-180}
BENCHMARK_TIMEOUT=${BENCHMARK_TIMEOUT_S:-900}
BENCHMARK_KILL_AFTER=${BENCHMARK_KILL_AFTER_S:-20}
RATES=${RATES:-"10 9 8 7 6 5 4 3 2 1"}
# RATES=${RATES:-"20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"}

case "$BENCHMARK_KILL_AFTER" in
    ''|*[!0-9]*|0) echo "BENCHMARK_KILL_AFTER_S must be a positive integer" >&2; exit 2 ;;
esac


mkdir -p "$LOG_DIR"

START=${1:-10}

for rate in $RATES; do
# for rate in $(seq $START -1 1); do
    echo "============================================"
    echo "[$(date)] Starting benchmark with request-rate=$rate"
    echo "============================================"

    for dataset in servegen; do
        for dataset_mode in mm-image; do
            echo "--------------------------------------------"
            echo "[$(date)] Running benchmark with dataset=$dataset and dataset_mode=$dataset_mode"
            echo "--------------------------------------------"

            mkdir -p "${LOG_DIR}/${dataset}/${dataset_mode}"
            unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
            set +e
            timeout --signal=TERM --kill-after="$BENCHMARK_KILL_AFTER" "$BENCHMARK_TIMEOUT" \
                env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
                python -u benchmark_serving_chat_req_rate.py \
                --dataset-type="$dataset" \
                --servegen-mode="$dataset_mode" \
                --servegen-duration 180 \
                --request-timeout-s "$REQUEST_TIMEOUT" \
                --port "$PORT" \
                --addr "$ADDR" \
                --request-rate "$rate" \
                --bypass-cache \
                --dump-dir "$LOG_DIR/${dataset}/${dataset_mode}" \
                2>&1 | tee "${LOG_DIR}/${dataset}/${dataset_mode}/${rate}.log"
            benchmark_status=${PIPESTATUS[0]}
            set -e
            if [ "$benchmark_status" -ne 0 ]; then
                echo "[$(date)] benchmark failed or timed out (status=$benchmark_status); continuing with next rate" >&2
            fi

        done
    done

    echo "[$(date)] Finished rate=$rate"

    if [ "$rate" -lt "$START" ]; then
        echo "Sleeping ${SLEEP_INTERVAL}s before next run..."
        sleep "$SLEEP_INTERVAL"
    fi
done

echo "============================================"
echo "[$(date)] All benchmarks completed!"
echo "============================================"
