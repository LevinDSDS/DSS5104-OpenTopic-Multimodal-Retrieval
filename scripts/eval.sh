#!/usr/bin/env bash
# Evaluate an AITR checkpoint.
#   Usage: bash scripts/eval.sh runs/flickr30k_bert/best.ckpt
set -euo pipefail
CKPT=${1:?"please pass a checkpoint path"}
python -u eval.py "${CKPT}"
