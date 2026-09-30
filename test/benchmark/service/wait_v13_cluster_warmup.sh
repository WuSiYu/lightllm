#!/usr/bin/env bash
set -u

LOG_FILE=${1:?usage: wait_v13_cluster_warmup.sh MASTER_LOG URL [TIMEOUT_S]}
URL=${2:?usage: wait_v13_cluster_warmup.sh MASTER_LOG URL [TIMEOUT_S]}
TIMEOUT_S=${3:-600}
POLL_S=${POLL_S:-2}
MAX_ATTEMPTS=$((TIMEOUT_S / POLL_S))

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
if [[ "$LOG_FILE" != /* ]]; then
    LOG_FILE="$REPO_ROOT/$LOG_FILE"
fi

for _ in $(seq 1 "$MAX_ATTEMPTS"); do
    prefill_count=$(grep -a -c 'mode: prefill url' "$LOG_FILE" 2>/dev/null || true)
    decode_count=$(grep -a -c 'mode: decode url' "$LOG_FILE" 2>/dev/null || true)
    if [ "$prefill_count" -ge 3 ] && [ "$decode_count" -ge 1 ]; then
        if WARMUP_LONG_INPUT_TOKENS=${WARMUP_LONG_INPUT_TOKENS:-16000} \
            timeout --signal=TERM --kill-after=10 "$TIMEOUT_S" \
            ./wait_warmup_mixed.sh --url "$URL"; then
            echo 'mixed warmup complete'
            exit 0
        fi
        echo 'mixed warmup timed out or failed' >&2
        exit 1
    fi
    sleep "$POLL_S"
done

echo 'Timed out waiting for all Prefill/Decode registrations; inspect master.log.' >&2
exit 1
