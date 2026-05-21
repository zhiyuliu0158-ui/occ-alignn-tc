#!/usr/bin/env bash
set -euo pipefail

python scripts/make_toy_dataset.py --output_dir data/toy
python scripts/train.py --config configs/small_cpu.yaml --data_csv data/toy/toy_data.csv --output_dir runs/smoke --device cpu
python scripts/evaluate.py --checkpoint runs/smoke/best.pt --data_csv data/toy/toy_data.csv --output_dir runs/smoke_eval --split test --device cpu
python scripts/predict.py --checkpoint runs/smoke/best.pt --data_csv data/toy/toy_data.csv --output_dir runs/smoke_predict --device cpu

test -f runs/smoke/best.pt
test -f runs/smoke/predictions_val.csv
test -f runs/smoke/predictions_test.csv
test -f runs/smoke_eval/predictions_test.csv
test -f runs/smoke_predict/predictions.csv
echo "Smoke test passed."
