#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SELECTOR=flex_tp_v11 exec "$SCRIPT_DIR/start-cluster3_70b_p22.4d4_mps_prefill_fake_decode.sh" "$@"
