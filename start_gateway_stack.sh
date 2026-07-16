#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export QWEN_TTS_MODEL="${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
export QWEN_TTS_HOST="${QWEN_TTS_HOST:-127.0.0.1}"
export QWEN_TTS_WORKERS=1
export QWEN_TTS_WORKER_LOCK_TIMEOUT="${QWEN_TTS_WORKER_LOCK_TIMEOUT:-5}"
export QWEN_TTS_BACKENDS="${QWEN_TTS_BACKENDS:-http://127.0.0.1:18011,http://127.0.0.1:18012,http://127.0.0.1:18013,http://127.0.0.1:18014}"
export QWEN_TTS_GATEWAY_HOST="${QWEN_TTS_GATEWAY_HOST:-0.0.0.0}"
export QWEN_TTS_GATEWAY_PORT="${QWEN_TTS_GATEWAY_PORT:-18003}"
export QWEN_TTS_GATEWAY_QUEUE_TIMEOUT="${QWEN_TTS_GATEWAY_QUEUE_TIMEOUT:-15}"
export PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export QWEN_TTS_BACKEND_STARTUP_TIMEOUT="${QWEN_TTS_BACKEND_STARTUP_TIMEOUT:-600}"

PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

IFS=',' read -r -a BACKEND_URLS <<< "$QWEN_TTS_BACKENDS"
for backend_url in "${BACKEND_URLS[@]}"; do
  port="${backend_url##*:}"
  export QWEN_TTS_PORT="$port"
  echo "Starting Qwen3-TTS backend on 127.0.0.1:${QWEN_TTS_PORT}"
  "$PYTHON_BIN" api_server.py &
  PIDS+=("$!")
done

deadline=$((SECONDS + QWEN_TTS_BACKEND_STARTUP_TIMEOUT))
for backend_url in "${BACKEND_URLS[@]}"; do
  echo "Waiting for ${backend_url}/health"
  until curl -fsS --max-time 2 "${backend_url}/health" >/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "Timed out waiting for ${backend_url}/health" >&2
      exit 1
    fi
    sleep 2
  done
done

echo "Starting Qwen3-TTS gateway on ${QWEN_TTS_GATEWAY_HOST}:${QWEN_TTS_GATEWAY_PORT}"
"$PYTHON_BIN" gateway_server.py &
PIDS+=("$!")

wait -n "${PIDS[@]}"
exit 1
