#!/bin/bash
# 50 个 client 并行，各自阻塞循环发送长度 ~200 的短请求
HOST_IP=$(hostname -i | awk '{print $1}')
URL="http://${HOST_IP}:60011/generate"

NUM_CLIENTS=100
PROMPT_LEN=3000   # 约 200 token 的短请求

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
RESET='\033[0m'

# 构造 ~200 token 的 prompt
PROMPT="Oh fuck, What AI?"
for i in $(seq 1 $((PROMPT_LEN - 4))); do PROMPT="${PROMPT} token"; done

PAYLOAD=$(printf '{"inputs": "%s", "parameters":{"max_new_tokens":1, "frequency_penalty":1}}' "$PROMPT")

echo ""
echo -e "${YELLOW}$URL${RESET} - ${NUM_CLIENTS} clients, prompt_len ~${PROMPT_LEN}"
echo ""

client() {
  local id=$1
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    local start end elapsed exit_code
    start=$(date +%s%N)
    curl -s -f "$URL" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD" >/dev/null 2>&1
    exit_code=$?
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    if [ $exit_code -eq 0 ]; then
      echo -e "  ${GREEN}[c${id} #${attempt}] OK${RESET} ${elapsed}ms"
    else
      echo -e "  ${RED}[c${id} #${attempt}] FAIL${RESET} ${elapsed}ms"
      sleep 1
    fi
  done
}

# 清理子进程
trap 'kill 0' SIGINT SIGTERM EXIT

for id in $(seq 1 $NUM_CLIENTS); do
  client "$id" &
done

wait
