#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

/usr/bin/env bash "$ROOT_DIR/scripts/run_trader_wsl.sh" --profile test --log-profile trader_test "$@"
