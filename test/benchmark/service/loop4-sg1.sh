#!/bin/bash
set -e

ADDR=$(hostname -i)
PORT=60011
SLEEP_INTERVAL=60
LOG_DIR="_/260408-70b_p22d4_v6.1_mps_sg1"

mkdir -p "$LOG_DIR"

START=${1:-20}

for rate in $(seq $START -1 1); do
    echo "============================================"
    echo "[$(date)] Starting benchmark with request-rate=$rate"
    echo "============================================"

    for dataset in servegen; do
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

    if [ "$rate" -lt "$START" ]; then
        echo "Sleeping ${SLEEP_INTERVAL}s before next run..."
        sleep "$SLEEP_INTERVAL"
    fi
done

echo "============================================"
echo "[$(date)] All benchmarks completed!"
echo "============================================"
