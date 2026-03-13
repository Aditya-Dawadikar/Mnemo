#!/bin/sh
set -eu

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

pick_gpu() {
  # Requires nvidia-smi; prints the index of the first available GPU.
  # Returns 1 if no GPU is found.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  echo "[ollama-startup] Listing available GPUs:"
  nvidia-smi -L 2>/dev/null || true

  # Iterate over each GPU index reported by nvidia-smi
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$gpu_count" -eq 0 ]; then
    return 1
  fi

  i=0
  while [ "$i" -lt "$gpu_count" ]; do
    status=$(nvidia-smi -i "$i" --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || true)
    if [ -n "$status" ]; then
      echo "[ollama-startup] Selected GPU $i: $status"
      echo "$i"
      return 0
    fi
    i=$((i + 1))
  done

  return 1
}

echo "[ollama-startup] Checking GPU availability"
SELECTED_GPU=$(pick_gpu) || SELECTED_GPU=""

if [ -n "$SELECTED_GPU" ]; then
  echo "[ollama-startup] GPU access confirmed, using GPU $SELECTED_GPU"
  export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
else
  echo "[ollama-startup] No GPU found"
  if [ "$REQUIRE_GPU" = "1" ]; then
    echo "[ollama-startup] REQUIRE_GPU=1, refusing to start without a GPU"
    exit 1
  fi
  echo "[ollama-startup] REQUIRE_GPU=0, continuing on CPU"
fi

echo "[ollama-startup] Starting ollama serve"
ollama serve &
OLLAMA_PID="$!"

cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
}

trap cleanup INT TERM

echo "[ollama-startup] Waiting for Ollama API to become available"
until ollama list >/dev/null 2>&1; do
  sleep 1
done

if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$MODEL"; then
  echo "[ollama-startup] Model '$MODEL' not found. Pulling now"
  ollama pull "$MODEL"
else
  echo "[ollama-startup] Model '$MODEL' already exists"
fi

echo "[ollama-startup] Ollama is ready with model '$MODEL'"
wait "$OLLAMA_PID"