#!/bin/bash
set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
#python -m pytest \
#    tests/test_rustbpe.py \
#    -v -s

python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

python -m pytest sandbox/test_tok_train.py -v -s
