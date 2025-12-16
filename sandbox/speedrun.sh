#!/bin/bash
set -x

# This script is a modified version of dev/runcpu.sh that makes playing around on a Macbook or Mac mini easier.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv

which nvidia-smi
NO_CUDA=$?

if ((NO_CUDA)); then
    uv sync --extra cpu
else
    uv sync --extra gpu
    NPROC_PER_NODE=8
fi

source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# wipe the report
python -m nanochat.report reset

set -e

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    python -m nanochat.dataset -n 8 # train tokenizer on ~1B characters
    python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!
    bash sandbox/tok_train.sh
    wait $DATASET_DOWNLOAD_PID
fi

# The following settings have been adjusted for training a model of similar depth and max_seq_len as speedrun.sh
# on a MPS machine (M4 Pro, 64GB).

MODEL_DEPTH=${MODEL_DEPTH:-20}
if [ -d "${NANOCHAT_BASE_DIR}/base_checkpoints/d${MODEL_DEPTH}" ]; then
    # Find last modified checkpoint dir
    LAST_STEP=$(jq -r .step "$(find "${NANOCHAT_BASE_DIR}/base_checkpoints/d${MODEL_DEPTH}" -type f -name "meta*" -print0 | xargs -0 ls -1t | head -1)")
else
    LAST_STEP=0
fi

BASE_TRAIN_ARGS=(
    --depth="$MODEL_DEPTH"
    --run="$WANDB_RUN"
)
if ((LAST_STEP)); then
    BASE_TRAIN_ARGS+=(--resume_from_step="$LAST_STEP")
fi
if ((NO_CUDA)); then
    # Requires: total_batch_size % (device_batch_size * max_seq_len * world_size) == 0
    # NOTE: world_size = 1 on cpu, 8 on 8XH100
    BASE_TRAIN_ARGS+=(
        --device_batch_size="${BASE_TRAIN_DEVICE_BATCH_SIZE:-2}"
        --total_batch_size="${BASE_TRAIN_TOTAL_BATCH_SIZE:-16384}"
        --eval_tokens="${BASE_TRAIN_EVAL_TOKENS:-4096}"
        --core_metric_max_per_task="${BASE_TRAIN_CORE_METRIC_MAX_PER_TASK:-12}"
        --save_every=5400
    )
    python -m scripts.base_train ${BASE_TRAIN_ARGS[*]}
    python -m scripts.base_loss --device_batch_size="${BASE_LOSS_DEVICE_BATCH_SIZE:-2}" --split_tokens=4096
    python -m scripts.base_eval --max-per-task=16
else
    BASE_TRAIN_ARGS+=(
        --save_every=1000
    )
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- ${BASE_TRAIN_ARGS[*]}
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_loss
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval
fi

# midtraining
if ! ((NO_CUDA)) || [ "$TRAIN_MID" == 'true' ]; then
    if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
        curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi
    if ((NO_CUDA)); then
        python -m scripts.mid_train \
            --device_batch_size="${MID_TRAIN_DEVICE_BATCH_SIZE:-2}" \
            --eval_every=50 \
            --eval_tokens="${MID_TRAIN_EVAL_TOKENS:-4096}" \
            --total_batch_size=4096 \
            --num_iterations=100
        python -m scripts.chat_eval -i=mid --max-new-tokens=128 --max-problems=20
    else
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
    fi
fi

# SFT
if ! ((NO_CUDA)) || [ "$TRAIN_SFT" == 'true' ]; then
    if ((NO_CUDA)); then
        python -m scripts.chat_sft \
            --device_batch_size=4 \
            --target_examples_per_step=32 \
            --num_iterations=10 \
            --eval_steps=4 \
            --eval_metrics_max_problems=16
        python -m scripts.chat_eval --source=sft --max-new-tokens=128 --max-problems=20
    else
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
    fi
fi

if ! ((NO_CUDA)); then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K
fi

python -m nanochat.report generate