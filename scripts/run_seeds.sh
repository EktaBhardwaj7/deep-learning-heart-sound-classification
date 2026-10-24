#!/usr/bin/env bash
# Repeat a configuration over several seeds (needed before claiming any difference). Usage:
#   SEEDS="42 43 44" bash scripts/run_seeds.sh configs/rq1_logmel.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:?config path}
for s in ${SEEDS:-42 43 44}; do
  name=$(python -c "import yaml;print(yaml.safe_load(open('$CFG')).get('project',{}).get('name','run'))")
  ov=("project.seed=$s" "project.name=${name}_s$s")
  python -m hsd train    --config "$CFG" --resume auto --set "${ov[@]}"
  python -m hsd evaluate --config "$CFG" --set "${ov[@]}"
done
