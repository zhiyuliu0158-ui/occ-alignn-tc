#!/bin/bash
#
#SBATCH --job-name=eval2
#SBATCH --output=e1_diff_out
#SBATCH --error=e1_diff_err
#SBATCH --nodes=1   ### 节点数
#SBATCH --gres=gpu:1 ### 使用GPU时不用指定核心数
#SBATCH --time=72:00:00
#SBATCH -A hmt03
#SBATCH -p h200

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Start time: $(date)"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Optional: activate a conda environment when submitting with:
#   sbatch --export=ALL,CONDA_ENV=pb-occ-alignn-full-tc scripts/eval2.sh
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "${HOME}/.bashrc"
  conda activate "${CONDA_ENV}"
fi

# Optional: override Python executable when submitting with:
#   sbatch --export=ALL,PYTHON=/path/to/python scripts/eval2.sh
PYTHON_BIN="${PYTHON:-python}"

# Required/overridable paths:
#   sbatch --export=ALL,CHECKPOINT=runs/real_exp/best.pt,DATA_CSV=/path/to/data.csv scripts/eval2.sh
CHECKPOINT="${CHECKPOINT:-runs/real_exp/best.pt}"
DATA_CSV="${DATA_CSV:-data/toy/toy_data.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/eval2}"
SPLIT="${SPLIT:-test}"

echo "Python: $(${PYTHON_BIN} --version)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data CSV: ${DATA_CSV}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Split: ${SPLIT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  echo "Example:" >&2
  echo "  sbatch --export=ALL,CHECKPOINT=runs/real_exp/best.pt,DATA_CSV=/path/to/data.csv,OUTPUT_DIR=runs/eval2,SPLIT=test scripts/eval2.sh" >&2
  exit 1
fi

if [[ ! -f "${DATA_CSV}" ]]; then
  echo "ERROR: data CSV not found: ${DATA_CSV}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/evaluate.py \
  --checkpoint "${CHECKPOINT}" \
  --data_csv "${DATA_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --device cuda

echo "End time: $(date)"
echo "Done."
