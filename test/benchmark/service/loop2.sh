#!/bin/bash
set -e

ADDR=$(hostname -i)
PORT=60011
SLEEP_INTERVAL=60
LOG_DIR="_"

mkdir -p "$LOG_DIR"

for rate in $(seq 20 -1 1); do
    echo "============================================"
    echo "[$(date)] Starting benchmark with request-rate=$rate"
    echo "============================================"

    python -u benchmark_serving_chat_req_rate.py \
        --dataset-type=simple.2 \
        --port "$PORT" \
        --addr "$ADDR" \
        --request-rate "$rate" \
        --bypass-cache \
        2>&1 | tee "${LOG_DIR}/70b_p22d4_v3.4_s2_${rate}.log"
        # 2>&1 | tee "${LOG_DIR}/70b_p22.4d4_v3.4_s2_${rate}.log"

    echo "[$(date)] Finished rate=$rate"

    if [ "$rate" -lt 20 ]; then
        echo "Sleeping ${SLEEP_INTERVAL}s before next run..."
        sleep "$SLEEP_INTERVAL"
    fi
done

echo "============================================"
echo "[$(date)] All benchmarks completed!"
echo "============================================"
