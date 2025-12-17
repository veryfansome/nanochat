#!/bin/bash
set -ex

uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
python -m pytest tests/test_rustbpe.py -v -s

python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
python -m pytest tests/test_tok_train.py -v -s
