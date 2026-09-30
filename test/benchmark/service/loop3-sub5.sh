#!/bin/bash
set -e

ADDR=$(hostname -i | awk '{print $1}')
PORT=60011
SLEEP_INTERVAL=60
LOG_DIR="_/260906-70b_p22.4d4_v12_mps_sche12_simple.1"
# LOG_DIR="_/260331-70b_p22.4d4_v4.1_mps_sche1_simple.1"

mkdir -p "$LOG_DIR"

for rate in $(seq 20 -1 1); do
    echo "============================================"
    echo "[$(date)] Starting benchmark with request-rate=$rate"
    echo "============================================"

    for dataset in simple.1-5; do
    # for dataset in simple.1-0 simple.1-3 simple.1-5 simple.1-10 simple.1-20; do
        echo "--------------------------------------------"
        echo "[$(date)] Running benchmark with dataset=$dataset"
        echo "--------------------------------------------"

        mkdir -p "${LOG_DIR}/${dataset}"
        python -u benchmark_serving_chat_req_rate.py \
            --dataset-type="$dataset" \
            --port "$PORT" \
            --addr "$ADDR" \
            --request-rate "$rate" \
            --bypass-cache \
            --dump-dir "$LOG_DIR/${dataset}" \
            2>&1 | tee "${LOG_DIR}/${dataset}/${rate}.log"

        sleep 20
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
