#!/usr/bin/env bash
# Full pipeline for one configuration on the real data (resumable). Usage: bash scripts/run_experiment.sh configs/joint_main.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/joint_main.yaml}
for c in prepare train evaluate robustness uncertainty explain export device-eval report; do
  if [ "$c" = "train" ]; then python -m hsd train --config "$CFG" --resume auto
  else python -m hsd "$c" --config "$CFG"; fi
done
