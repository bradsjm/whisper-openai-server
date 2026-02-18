#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ACCELERATION="${WHISPER_ACCELERATION:-metal}"

export HOST
export PORT
export WHISPER_ACCELERATION="$ACCELERATION"

exec cargo run --release -- "$@"
