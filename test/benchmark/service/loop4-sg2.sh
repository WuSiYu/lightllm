#!/bin/bash
set -e

ADDR=$(hostname -i)
PORT=60011
SLEEP_INTERVAL=20
LOG_DIR="_/260409-70b_p22.4d4_v8_mps_flex_naive_8k_sg1"

mkdir -p "$LOG_DIR"

START=${1:-20}

# for rate in $(seq $START -1 1); do
for rate in 15 12 10 20 19 18 17 16 14 13 11 9 8 5 1; do
    echo "============================================"
    echo "[$(date)] Starting benchmark with request-rate=$rate"
    echo "============================================"

    for dataset in servegen; do
        for dataset_mode in m-large; do
        # for dataset_mode in mm-image m-large; do
            echo "--------------------------------------------"
            echo "[$(date)] Running benchmark with dataset=$dataset and dataset_mode=$dataset_mode"
            echo "--------------------------------------------"

            mkdir -p "${LOG_DIR}/${dataset}/${dataset_mode}"
            python -u benchmark_serving_chat_req_rate.py \
                --dataset-type="$dataset" \
                --servegen-mode="$dataset_mode" \
                --port "$PORT" \
                --addr "$ADDR" \
                --request-rate "$rate" \
                --bypass-cache \
                --dump-dir "$LOG_DIR/${dataset}/${dataset_mode}" \
                2>&1 | tee "${LOG_DIR}/${dataset}/${dataset_mode}/${rate}.log"

            sleep 10
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
