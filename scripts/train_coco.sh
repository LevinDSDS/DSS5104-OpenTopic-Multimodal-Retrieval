#!/usr/bin/env bash
# Train AITR on MS-COCO.
#   Usage: bash scripts/train_coco.sh configs/coco_bert.yaml
set -euo pipefail
CONFIG=${1:-configs/coco_bert.yaml}
python -u train.py --config "${CONFIG}"
