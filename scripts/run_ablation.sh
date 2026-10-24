#!/usr/bin/env bash
# Multi-dataset ablation (proposal configs A, B, C, E). D = A and B reported side by side.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  local name=$1; shift
  local ov=("project.name=ablation_$name" "$@")
  python -m hsd train    --config configs/joint_main.yaml --resume auto --set "${ov[@]}"
  python -m hsd evaluate --config configs/joint_main.yaml --set "${ov[@]}"
}
run A_yaseen_only  data.tasks=[yaseen]
run B_circor_only  data.tasks=[circor]
run C_joint        data.tasks=[yaseen,circor]
run E_joint_frozen data.tasks=[yaseen,circor] train.freeze_epochs=100000
