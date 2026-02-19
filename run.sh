#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ACCELERATION="${WHISPER_ACCELERATION:-metal}"

# Auto-detect platform for feature flags
if [[ "$OSTYPE" == "darwin"* ]]; then
    FEATURES="--features metal"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    FEATURES="--features cuda"
else
    # Unknown platform - build without features (CPU-only)
    FEATURES=""
fi

export HOST
export PORT
export WHISPER_ACCELERATION="$ACCELERATION"

exec cargo run --release $FEATURES -- "$@"
