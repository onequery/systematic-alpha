#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

print_script_help() {
  cat <<'EOF'
Usage:
  ./run_main.sh [main.py args...]

Examples:
  ./run_main.sh --collect-seconds 600 --final-picks 3
  ./run_main.sh --universe-file ./codes.txt --output-json ./out/picks.json
  ./run_main.sh --help

Common args:
  --key-file PATH
  --universe-file PATH
  --collect-seconds INT
  --max-symbols-scan INT
  --pre-candidates INT
  --final-picks INT
  --min-change-pct FLOAT
  --min-gap-pct FLOAT
  --min-prev-turnover FLOAT
  --min-strength FLOAT
  --min-vol-ratio FLOAT
  --min-bid-ask-ratio FLOAT
  --min-pass-conditions INT
  --min-maintain-ratio FLOAT
  --rest-sleep FLOAT
  --mock
  --user-id TEXT
  --output-json PATH

Tip:
  Sensitive values should be stored in .env.
EOF
}

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
else
  echo "[warn] .env not found. Falling back to current environment." >&2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${1:-}" == "--help-script" ]]; then
  print_script_help
  exit 0
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_script_help
  echo
fi

echo "[run] $PYTHON_BIN main.py $*"
exec "$PYTHON_BIN" main.py "$@"
