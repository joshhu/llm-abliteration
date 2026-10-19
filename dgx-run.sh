#!/usr/bin/env bash
# Abliterate Qwen/Qwen3.8-27B on a DGX Spark (GB10) host using heretic.
# Run these subcommands ON the DGX host (or via ssh).
#
#   ./dgx-run.sh start     # create the GPU container (sleep infinity)
#   ./dgx-run.sh deps      # pip install heretic-llm inside the container
#   ./dgx-run.sh download  # hf download the base model (~52 GB)
#   ./dgx-run.sh run       # start the headless abliteration in the background
#   ./dgx-run.sh status    # tail progress + trial checkpoint count
#   ./dgx-run.sh summary   # regenerate results/best-trial.md from the journal
#   ./dgx-run.sh watch     # when the run finishes, refresh results/best-trial.md
#   ./dgx-run.sh stop      # stop+remove the container (keeps downloads/output)
#
# Typical one-shot: ./dgx-run.sh start && ./dgx-run.sh deps && ./dgx-run.sh download && ./dgx-run.sh run

set -euo pipefail

IMAGE=${IMAGE:-vllm/vllm-openai:v0.27.1}          # arm64 + CUDA 13 + transformers 5.x (qwen3_5 support)
NAME=${NAME:-heretic-qwen38}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B}
WORK_HOST=${WORK_HOST:-"$HOME/heretic/qwen3.8-27b"}   # -> /heretic/qwen3.8-27b (must contain config.toml + run_heretic.py)
ROOT_HOST=${ROOT_HOST:-"$HOME/heretic"}               # -> /heretic
SAVE_DIR=${SAVE_DIR:-/heretic/out/Qwen3.8-27B-abliterated}
HF_CACHE=${HF_CACHE:-"$HOME/.cache/huggingface"}

ensure_workdir() {
  [ -f "$WORK_HOST/config.toml" ] || { echo "missing $WORK_HOST/config.toml"; exit 1; }
  [ -f "$WORK_HOST/run_heretic.py" ] || { echo "missing $WORK_HOST/run_heretic.py"; exit 1; }
  mkdir -p "$ROOT_HOST/out"
}

case "${1:-}" in
  start)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=16g \
      -v "$HF_CACHE":/root/.cache/huggingface \
      -v "$ROOT_HOST":/heretic \
      --entrypoint bash "$IMAGE" -c 'sleep infinity'
    sleep 3
    docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
    ;;
  deps)
    docker exec "$NAME" bash -lc 'pip install -U heretic-llm 2>&1 | tail -3
      python3 -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))"'
    ;;
  download)
    docker exec "$NAME" bash -lc "hf download $BASE_MODEL"
    ;;
  run)
    ensure_workdir
    docker exec -d -e HERETIC_SAVE_DIR="$SAVE_DIR" "$NAME" bash -lc \
      "cd /heretic/qwen3.8-27b && python3 run_heretic.py $BASE_MODEL < /dev/null > run.log 2>&1; echo EXIT=\$? >> run.log"
    echo "started; tail with: ./dgx-run.sh status"
    ;;
  status)
    tr '\r' '\n' < "$WORK_HOST/run.log" | grep -viE 'it/s\]|s/it\]' | tail -15
    echo "--- trials checkpointed ---"
    wc -l "$WORK_HOST"/checkpoints/*.jsonl 2>/dev/null || echo "(none yet)"
    echo "--- output ---"
    ls -la "$ROOT_HOST/out" 2>/dev/null || true
    ;;
  summary)
    ensure_workdir
    mkdir -p "$WORK_HOST/results"
    docker exec "$NAME" bash -lc \
      "cd /heretic/qwen3.8-27b && python3 tools/summarize.py checkpoints/*.jsonl > results/best-trial.md"
    echo "wrote $WORK_HOST/results/best-trial.md"
    ;;
  watch)
    setsid nohup bash -c "
      while ! grep -q '^EXIT=' '$WORK_HOST/run.log' 2>/dev/null; do sleep 120; done
      mkdir -p '$WORK_HOST/results'
      docker exec '$NAME' bash -lc 'cd /heretic/qwen3.8-27b && python3 tools/summarize.py checkpoints/*.jsonl > results/best-trial.md' \
        && touch '$WORK_HOST/results/FINISHED'
    " >/dev/null 2>&1 </dev/null &
    echo "watcher started (regenerates results/best-trial.md when the run ends)"
    ;;
  stop)
    docker rm -f "$NAME" >/dev/null && echo "removed $NAME"
    ;;
  *)
    sed -n '2,17p' "$0"
    exit 1
    ;;
esac
