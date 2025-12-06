#!/bin/bash
set -x

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash sandbox/modified-runcpu.sh

# NOTE: This script is a modified version of dev/runcpu.sh that makes playing around on a Macbook or Mac mini easier.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# The following settings have been adjusted for training a model of similar depth and max_seq_len as speedrun.sh
# on a MPS machine (M4 Pro, 64GB).

# Requires: TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * WORLD_SIZE) == 0
MODEL_DEPTH=${MODEL_DEPTH:-20}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-2}
TOTAL_BATCH_SIZE=${total_batch_size:-16384}
EVAL_TOKENS=${EVAL_TOKENS:-4096}
CORE_METRIC_MAX_PER_TASK=${CORE_METRIC_MAX_PER_TASK:-12}
STEPS_PER_RUN=100

if [ -d "${NANOCHAT_BASE_DIR}/base_checkpoints/d${MODEL_DEPTH}" ]; then
    # Find last modified checkpoint dir
    LAST_STEP=$(jq -r .step "$(find "${NANOCHAT_BASE_DIR}/base_checkpoints/d${MODEL_DEPTH}" -type f -name "meta*" -print0 | xargs -0 ls -1t | head -1)")
    NUM_ITERATIONS=$((LAST_STEP + STEPS_PER_RUN))
else
    LAST_STEP=0
    NUM_ITERATIONS=$STEPS_PER_RUN
fi

# wipe the report
python -m nanochat.report reset

set -e

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
    python -m pytest tests/test_rustbpe.py -v -s
    python -m nanochat.dataset -n 8 # train tokenizer on ~1B characters
    python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!
    python -m sandbox.seed_tokens
    python -m scripts.tok_train --max_chars=2000000000 --seed_tokens "${NANOCHAT_BASE_DIR}/seed_tokens.json"
    python -m scripts.tok_eval
    wait $DATASET_DOWNLOAD_PID
fi

# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
BASE_TRAIN_ARGS=(
    --depth="$MODEL_DEPTH"
    --max_seq_len="$MAX_SEQ_LEN"
    --device_batch_size="$DEVICE_BATCH_SIZE"
    --total_batch_size="$TOTAL_BATCH_SIZE"
    --eval_tokens="$EVAL_TOKENS"
    --core_metric_max_per_task="$CORE_METRIC_MAX_PER_TASK"
    --num_iterations="$NUM_ITERATIONS"
)
if [ "$LAST_STEP" -gt 0 ]; then
  BASE_TRAIN_ARGS+=(--resume_from_step="$LAST_STEP")
fi
python -m scripts.base_train ${BASE_TRAIN_ARGS[*]}
python -m scripts.base_loss --device_batch_size="$DEVICE_BATCH_SIZE" --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# midtraining
if [ "$TRAIN_MID" == 'true' ]; then
    if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
        curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi
    python -m scripts.mid_train \
        --max_seq_len="$MAX_SEQ_LEN" \
        --device_batch_size="$DEVICE_BATCH_SIZE" \
        --eval_every=50 \
        --eval_tokens="$EVAL_TOKENS" \
        --total_batch_size=4096 \
        --num_iterations=100
    # eval results will be terrible, this is just to execute the code paths.
    # note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
    python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20
fi

# SFT
if [ "$TRAIN_SFT" == 'true' ]; then
    python -m scripts.chat_sft \
        --device_batch_size=4 \
        --target_examples_per_step=32 \
        --num_iterations=10 \
        --eval_steps=4 \
        --eval_metrics_max_problems=16
    python -m scripts.chat_eval --source=sft --max-new-tokens=128 --max-problems=20
fi

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
