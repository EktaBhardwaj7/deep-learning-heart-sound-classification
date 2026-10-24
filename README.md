# heart_screen v2

A screening aid that listens to heart sounds (phonocardiograms) and gives a health worker one of three signals:
**looks normal**, **possible murmur**, or **needs specialist review**. It is a research prototype for preliminary
screening in primary health centres. It does not make a diagnosis and has not been clinically validated.

Two public datasets are used: **Yaseen** (five diagnostic classes) and **CirCor DigiScope** (murmur present or
absent). They answer different questions, so one shared network has two output heads.

---

## 1. Data layout

Put your data in `data/` next to this project, exactly as you have it:

```
data/
├── heart_sound_dataset/                  Yaseen, 8 kHz
│   ├── AS/AS_New_3주기/New_AS_001.wav
│   ├── MR/...   MS/...   MVP/...   N/...
│   └── source/                           ignored (original .rar archives)
└── circor/
    ├── training_data.csv                 metadata: Patient ID, Murmur, Murmur locations, ...
    ├── LICENSE.txt  RECORDS  SHA256SUMS.txt
    └── training_data/training_data/
        ├── 2530_AV.wav   2530_AV.tsv   2530_AV.hea   (4 sites: AV, PV, TV, MV)
        └── 2530.txt                      patient record
```

The code searches for CirCor `.wav` files recursively, so the nested `training_data/training_data` folder is found
automatically. Check your data before training:

```bash
python scripts/check_data.py --config configs/joint_main.yaml
```

It prints `READY` or lists what is missing.

## 2. Install (Ubuntu)

```bash
cd ~/heart_screen                       # this project
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt         # install torch from pytorch.org instead if you use an NVIDIA GPU
python -m pytest -q tests               # 27 tests, about 10 seconds
bash scripts/verify.sh                  # full pipeline on synthetic data: should print ALL STAGES PASSED
```

`verify.sh` uses synthetic audio in a temporary folder and never touches `data/`. It proves the code runs. It says
nothing about the science.

## 3. Run on your data

```bash
python -m hsd prepare    --config configs/joint_main.yaml      # index, labels, leakage-free splits
python -m hsd train      --config configs/joint_main.yaml      # resumes automatically if interrupted
python -m hsd evaluate   --config configs/joint_main.yaml      # metrics, figures, operating point
python -m hsd robustness --config configs/joint_main.yaml      # noise: white, pink, 50 Hz hum
python -m hsd uncertainty --config configs/joint_main.yaml     # Monte Carlo dropout, triage zones
python -m hsd explain    --config configs/joint_main.yaml      # Grad-CAM, occlusion, faithfulness
python -m hsd export     --config configs/joint_main.yaml      # ONNX fp32 and int8, parity checks
python -m hsd device-eval --config configs/joint_main.yaml     # results from the exported files alone
python -m hsd report     --config configs/joint_main.yaml      # runs/joint_main/REPORT.md
```

Or everything in one go: `python -m hsd all --config configs/joint_main.yaml`

Any setting can be changed from the command line:

```bash
python -m hsd train --config configs/joint_main.yaml --set train.epochs=40 model.name=crnn model.pooling=attention
```

Outputs go to `runs/<name>/` (metrics, figures, report, `run.log`, `config_resolved.yaml`) and
`checkpoints/<name>/` (`best.pth`, `last.pth`).

## 4. Experiments (proposal mapping)

| Research question | Config | Command |
|---|---|---|
| RQ1, raw vs log-mel (matched size) | `rq1_logmel.yaml`, `rq1_raw.yaml` | `SEEDS="42 43 44" bash scripts/run_seeds.sh configs/rq1_logmel.yaml` |
| RQ2, joint learning (A, B, C, E) | `joint_main.yaml` | `bash scripts/run_ablation.sh` |
| RQ3, temporal models and attention | `rq3_crnn_attention.yaml`, `rq3_crnn_mean.yaml` | `bash scripts/run_experiment.sh configs/rq3_crnn_attention.yaml` |
| RQ4, uncertainty | `joint_main.yaml` | `python -m hsd uncertainty ...` |
| RQ5, explanations | `joint_main.yaml` | `python -m hsd explain ...` |

Backbone sizes (checked by `python -m hsd params --config ...` and by a test):

| Model | Backbone parameters |
|---|---|
| RQ1 log-mel: `cnn2d`, base width 32 | 582,304 |
| RQ1 raw: `resnet1d`, base width 24 | 543,096 (6.7% smaller) |
| Main joint model: `resnet2d`, base width 32 | 2,795,744 |

## 5. How the pipeline makes its decisions

Each rule below is fixed before the test set is used, because a result that depends on the test set is not a result.

- **Labels (CirCor).** The murmur belongs to the patient, but it is audible only at some sites. A recording is
  Present only if its site is listed in `Murmur locations`. Other recordings of a murmur patient follow
  `nonmurmur_site_policy` (`absent` by default). Patients who are Present with no listed site are excluded.
  Patients with murmur status Unknown are held out and used only for the uncertainty analysis.
- **Splits.** Yaseen is split by recording (it has no patient IDs). CirCor is split by patient. A leakage check
  stops the run if any patient or recording appears in two partitions. Splits are saved and reused, so every
  model sees the same partition.
- **Preprocessing.** Resample to 4 kHz, zero-phase band-pass 25 to 1,000 Hz, 5-second windows, log-mel with 64
  bands. Each window is standardised. Training and the device call the same function (`hsd/audio.py`), and a test
  checks that they produce the same features.
- **Recording rule.** How windows are combined into one decision (mean, max, median, noisy-OR, vote) is chosen
  on validation data. Noisy-OR saturates when a recording has many windows, which was the main fault in v1.
- **Operating point (CirCor).** The threshold that gives at least 90% sensitivity is chosen on validation and then
  applied once to the test set.
- **Calibration.** Temperature scaling is fitted on validation, and only when there are at least 300 validation
  windows. Below that it is skipped, and the report says why.
- **Training.** The backbone is frozen for 3 epochs, then released at one tenth of the head learning rate (v1
  spiked at the release). Weights are averaged with an exponential moving average, and evaluation uses the averaged
  weights. Early stopping uses 0.4 × Yaseen macro-F1 + 0.6 × CirCor macro-F1.
- **Triage zones.** The discharge and refer thresholds are chosen on validation to reach a negative predictive
  value of 0.95 and a positive predictive value of 0.80. The fallback values are used only if the targets cannot be
  reached, and the report says so.
- **Confidence intervals.** For CirCor, the bootstrap resamples patients, not windows, because windows from one
  child are not independent.

## 6. Device

```bash
python -m hsd infer --model runs/joint_main/export/model_int8.onnx \
                    --meta runs/joint_main/export/meta.json --task circor --wav some_recording.wav
```

The device runs the exported model offline. The metadata file carries the spectrogram settings, the rule, the
temperature and the operating threshold, so the device cannot drift from the training setup. `device-eval` reports
how much the int8 model loses compared with fp32, which is the on-device accuracy check from the proposal.

Monte Carlo dropout (30 passes) is for the desktop analysis. On the Raspberry Pi Zero 2 W, measure the latency first
and use fewer passes or a single pass if needed.

## 7. What changed from v1

| Issue in v1 | Fix in v2 |
|---|---|
| Noisy-OR saturated; recording-level CirCor F1 fell to 0.23 | Recording rule chosen on validation; test never used to choose it |
| Temperature scaling fitted on 100 validation recordings made calibration worse | Only with at least 300 validation windows |
| Validation loss spiked when the backbone was released | Backbone learning rate × 0.1 after release |
| Training and device preprocessing could differ slightly (torch vs numpy standard deviation) | One shared `standardise()`, tested |
| Deletion ratio could be negative and meaningless | Reported as undefined when removing random regions does not hurt |
| RQ1 Log-Mel model was not matched to the raw model | `cnn2d` 582,304 vs `resnet1d` 543,096 |
| Triage thresholds were fixed numbers | Chosen on validation to hit NPV and PPV targets |
| No way to check the data before a long run | `scripts/check_data.py` |
| Some results could not be reproduced | Splits saved and reused; config saved with every run; seeds in `run_seeds.sh` |

## 8. Limits, and what needs your input

- **CirCor `.tsv` state codes are assumed** (1 = S1, 2 = systole, 3 = S2, 4 = diastole). Check them against the dataset
  readme. The `.tsv` files give cardiac phase only, not murmur extents, so the phase analysis cannot show that an
  explanation matches a murmur.
- **Results on real data are not yet known.** Every number in this project so far is from a single run on public
  data or from synthetic audio. Repeat each comparison over at least three seeds before claiming a difference.
- **The Yaseen test set is small** (about 100 recordings, roughly 20 per class). Its intervals are wide.
- **The DIY chest piece has not been tested** against a reference recorder. Objective 8 of the proposal covers this.
- **Regulation.** A device that gives screening results may need CDSCO classification before a field pilot.
- **Safety.** Lithium cells need protection circuits. Do not charge unattended.

## 9. Project layout

```
configs/        base.yaml + experiment files (they extend base.yaml; only differences are written)
hsd/
  cli.py        commands          config.py    YAML, inheritance, overrides
  labels.py     recording index   splits.py    leakage-free splits
  audio.py      signal processing (shared with the device)
  dataset.py    window dataset    pipeline.py  glue: index -> windows -> loaders
  models.py     backbones, pooling, two-head network
  trainer.py    training, EMA, freezing, resume
  evaluate.py   aggregation, operating point, calibration, per-task evaluation
  metrics.py    metrics and bootstrap   uncertainty.py  MC dropout, triage
  robustness.py noise tests            explain.py   Grad-CAM, occlusion, faithfulness
  export.py     ONNX, int8, parity, device inference
  report.py     REPORT.md builder     plots.py     figures
scripts/        verify.sh, check_data.py, make_synthetic.py, run_seeds.sh, run_ablation.sh, run_experiment.sh
tests/          27 tests: config, labels, splits, metrics, models, audio, resume, export parity, device features
```
