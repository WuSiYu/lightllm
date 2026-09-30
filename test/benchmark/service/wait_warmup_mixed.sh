#!/usr/bin/env bash
set -u

URL="http://127.0.0.1:60011/generate"
MAX_ATTEMPTS="${WARMUP_MAX_ATTEMPTS:-120}"
WARMUP_TIMEOUT_S="${WARMUP_TIMEOUT_S:-600}"
WARMUP_LONG_INPUT_TOKENS="${WARMUP_LONG_INPUT_TOKENS:-4096}"
SHORT_PAYLOAD='{"inputs":"warmup short token","parameters":{"max_new_tokens":16,"ignore_eos":true}}'
LONG_PAYLOAD="$(WARMUP_LONG_INPUT_TOKENS="$WARMUP_LONG_INPUT_TOKENS" python - <<'PY'
import json
import os
length = int(os.environ["WARMUP_LONG_INPUT_TOKENS"])
print(json.dumps({"inputs": "token " * length, "parameters": {"max_new_tokens": 16, "ignore_eos": True}}))
PY
)"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$MAX_ATTEMPTS" in
    ''|*[!0-9]*) echo "--max-attempts must be an integer" >&2; exit 2 ;;
esac
case "$WARMUP_TIMEOUT_S" in
    ''|*[!0-9]*) echo "WARMUP_TIMEOUT_S must be an integer number of seconds" >&2; exit 2 ;;
esac
case "$WARMUP_LONG_INPUT_TOKENS" in
    ''|*[!0-9]*|0) echo "WARMUP_LONG_INPUT_TOKENS must be a positive integer" >&2; exit 2 ;;
esac

short_ok=0
long_ok=0
start_time=$(date +%s)
deadline=$((start_time + WARMUP_TIMEOUT_S))
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        break
    fi
    if [ "$short_ok" -lt 3 ] && curl --max-time 30 -sSf "$URL" \
        -H 'Content-Type: application/json' -d "$SHORT_PAYLOAD" >/dev/null 2>&1; then
        short_ok=$((short_ok + 1))
        echo "warmup short $short_ok/3 (attempt $attempt)"
    else
        [ "$short_ok" -ge 3 ] || short_ok=0
    fi
    if [ "$long_ok" -lt 3 ] && curl --max-time 60 -sSf "$URL" \
        -H 'Content-Type: application/json' -d "$LONG_PAYLOAD" >/dev/null 2>&1; then
        long_ok=$((long_ok + 1))
        echo "warmup long $long_ok/3 (attempt $attempt)"
    else
        [ "$long_ok" -ge 3 ] || long_ok=0
    fi
    if [ "$short_ok" -ge 3 ] && [ "$long_ok" -ge 3 ]; then
        echo "mixed warmup complete"
        exit 0
    fi
    sleep 2
done

elapsed=$(( $(date +%s) - start_time ))
echo "mixed warmup timed out after ${elapsed}s and $MAX_ATTEMPTS attempts" >&2
exit 1
