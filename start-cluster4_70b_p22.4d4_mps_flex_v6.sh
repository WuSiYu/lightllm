#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLEX_TP_VERSION=v6 exec "$SCRIPT_DIR/start-cluster4_70b_p22.4d4_mps_flex_v3.sh" "$@"
