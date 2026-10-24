#!/usr/bin/env bash
# Proof that the whole pipeline runs. Uses SYNTHETIC audio in a temporary folder and never touches your data.
# It checks the code, not the science.   bash scripts/verify.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
step() {
  local name=$1; shift
  if "$@" >"$T/$name.log" 2>&1; then
    echo "PASS  $name"
  else
    echo "FAIL  $name   (log: $T/$name.log)"; tail -15 "$T/$name.log"; fail=1
  fi
}
step synthetic_data $PY scripts/make_synthetic.py --root "$T/data"
step unit_tests     $PY -m pytest -q tests

COMMON="data.root=$T/data project.runs_dir=$T/runs project.checkpoint_dir=$T/ckpt data.cache_dir=cache \
 train.num_workers=0 model.base_channels=8 model.hidden_dim=16 train.batch_size.yaseen=8 \
 train.batch_size.circor=8 train.epochs=2 train.freeze_epochs=1 train.warmup_epochs=1 eval.bootstrap=20 \
 eval.mc_samples=3 eval.temperature_min_windows=50 eval.explain.max_samples=4 eval.explain.figures=1 \
 eval.noise_snrs_db=[10] eval.noise_kinds=[white] project.device=cpu"

for cfg in joint_main rq1_logmel rq1_raw rq3_crnn_attention; do
  ov="$COMMON"
  step "${cfg}_prepare"     $PY -m hsd prepare     --config configs/$cfg.yaml --set $ov
  step "${cfg}_train"       $PY -m hsd train       --config configs/$cfg.yaml --set $ov
  step "${cfg}_resume"      $PY -m hsd train       --config configs/$cfg.yaml --resume auto --set $ov train.epochs=3
  step "${cfg}_evaluate"    $PY -m hsd evaluate    --config configs/$cfg.yaml --set $ov
  step "${cfg}_robustness"  $PY -m hsd robustness  --config configs/$cfg.yaml --set $ov
  step "${cfg}_uncertainty" $PY -m hsd uncertainty --config configs/$cfg.yaml --set $ov
  step "${cfg}_explain"     $PY -m hsd explain     --config configs/$cfg.yaml --set $ov
  step "${cfg}_export"      $PY -m hsd export      --config configs/$cfg.yaml --set $ov
  step "${cfg}_device_eval" $PY -m hsd device-eval --config configs/$cfg.yaml --set $ov
  step "${cfg}_report"      $PY -m hsd report      --config configs/$cfg.yaml --set $ov
done

if [ $fail -eq 0 ]; then echo "ALL STAGES PASSED"; else echo "SOME STAGES FAILED"; fi
exit $fail
