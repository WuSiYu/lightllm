#!/bin/bash
HOST_IP=$(hostname -i)
URL="http://${HOST_IP}:60011/generate"
SUCCESS=0
ATTEMPT=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
BG_GREEN='\033[42;1;37m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}Warmup${RESET} ${YELLOW}$URL${RESET} (5 consecutive successes)"
echo ""

while true; do
  ATTEMPT=$((ATTEMPT + 1))

  START=$(date +%s%N)
  CONTENT=$(curl -s -f "$URL" \
    -H "Content-Type: application/json" \
    -d '{"inputs": "Oh fuck, What AI?", "parameters":{"max_new_tokens":64, "frequency_penalty":1}}' 2>/dev/null)
  EXIT_CODE=$?
  END=$(date +%s%N)
  ELAPSED=$(( (END - START) / 1000000 ))

  BAR=""
  if [ $EXIT_CODE -eq 0 ]; then
    SUCCESS=$((SUCCESS + 1))
    for i in 1 2 3 4 5; do
      if [ $i -le $SUCCESS ]; then BAR="${BAR}${GREEN}#${RESET}"; else BAR="${BAR}."; fi
    done
    echo -e "\r\033[K  ${BOLD}#${ATTEMPT}${RESET} ${GREEN}OK${RESET} ${ELAPSED}ms [${BAR}] ${SUCCESS}/5 - ${CONTENT}"

    if [ $SUCCESS -eq 5 ]; then
      echo ""
      echo -e "${BG_GREEN}                                    ${RESET}"
      echo -e "${BG_GREEN}                                    ${RESET}"
      echo -e "${BG_GREEN}     ▗▄▄▖ ▗▄▄▄▖ ▗▄▖ ▗▄▄▄ ▗▖  ▗▖     ${RESET}"
      echo -e "${BG_GREEN}     ▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌  █ ▝▚▞▘      ${RESET}"
      echo -e "${BG_GREEN}     ▐▛▀▚▖▐▛▀▀▘▐▛▀▜▌▐▌  █  ▐▌       ${RESET}"
      echo -e "${BG_GREEN}     ▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌▐▙▄▄▀  ▐▌       ${RESET}"
      echo -e "${BG_GREEN}                                    ${RESET}"
      echo -e "${BG_GREEN}                                    ${RESET}"
      echo ""

      # bell
      for i in {1..5}; do
        echo -ne "\a"
        sleep 0.2
      done
    fi

    # every 10 success, test long
    if [ $SUCCESS -gt 0 ] && [ $((SUCCESS % 10)) -eq 0 ]; then
      python test_long_request.py -f benchmark_serving_chat_req_rate.py -d 2 -t 2
    fi

  else
    SUCCESS=0
    printf "\r\033[K  ${BOLD}#${ATTEMPT}${RESET} ${RED}FAIL${RESET} ${ELAPSED}ms [.....] waiting..."
    sleep 2
  fi

done






