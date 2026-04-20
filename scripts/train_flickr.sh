#!/usr/bin/env bash
# Train AITR on Flickr30K.
#   Usage: bash scripts/train_flickr.sh configs/flickr30k_bert.yaml
set -euo pipefail
CONFIG=${1:-configs/flickr30k_bert.yaml}
python -u train.py --config "${CONFIG}"
