#!/bin/bash
# 顺序测试不同长度长请求的延迟：4001 ~ 16001，间隔 2000
# 使用 test_long_request.py 手动生成指定长度的 prompt

cd "$(dirname "$0")"

START_LEN=4001
END_LEN=32001
STEP=4000

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}Long request latency sweep${RESET} (${START_LEN} ~ ${END_LEN}, step ${STEP})"
echo ""

for len in $(seq $START_LEN $STEP $END_LEN); do
  echo -e "${YELLOW}===== prompt_length = ${len} =====${RESET}"
  python test_long_request.py -l "$len" -d 1 "$@"; sleep 1
  python test_long_request.py -l "$len" -d 1 "$@"; sleep 1
  python test_long_request.py -l "$len" -d 1 "$@"; sleep 1
  python test_long_request.py -l "$len" -d 1 "$@"; sleep 1
  python test_long_request.py -l "$len" -d 1 "$@"; sleep 1
  echo ""
done

echo -e "${GREEN}Done.${RESET}"
