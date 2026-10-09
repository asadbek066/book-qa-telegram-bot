#!/usr/bin/env bash
# Regenerate requirements.lock from requirements.txt and verify the two agree.
#
# This script is the single source of truth for the compile command. The header
# that uv writes into requirements.lock omits the index flags, so they live here.
#
# Usage: scripts/relock.sh [--compile-only | --check-only | --help]
#   (default)       compile the lock, then run the agreement check
#   --compile-only  compile the lock only
#   --check-only    run only the agreement check (same command as CI)
#
# The agreement check is `pip install --dry-run --no-index`, so it needs an
# environment that already has the locked packages (including the CPU torch
# wheel) installed. After a relock that changed versions, install the new lock
# (see README) and run `scripts/relock.sh --check-only`.
#
# Environment: PYTHON (default: python) selects the interpreter used for the check.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="all"
case "${1:-}" in
  "") ;;
  --compile-only) mode="compile" ;;
  --check-only) mode="check" ;;
  -h | --help)
    sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "relock.sh: unknown argument: $1 (try --help)" >&2
    exit 2
    ;;
esac

compile_lock() {
  if ! command -v uv > /dev/null 2>&1; then
    echo "relock.sh: 'uv' is not installed. Install it (https://docs.astral.sh/uv/ or 'pip install uv') and retry." >&2
    exit 1
  fi
  uv pip compile requirements.txt \
    --python-version 3.11 --universal \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --output-file requirements.lock
}

check_agreement() {
  # Identical to the CI step "Verify source requirements agree with the lock".
  if ! "${PYTHON:-python}" -m pip install --dry-run --no-index \
    -r requirements.txt -r requirements.lock; then
    echo "relock.sh: requirements.txt and requirements.lock disagree, or the active environment lacks the locked packages." >&2
    echo "relock.sh: install the lock (README, 'Updating dependencies') and re-run with --check-only." >&2
    exit 1
  fi
}

[ "$mode" = "check" ] || compile_lock
[ "$mode" = "compile" ] || check_agreement
