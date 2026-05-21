#!/bin/bash
#
#SBATCH --job-name=train_tc
#SBATCH --output=train_tc_out
#SBATCH --error=train_tc_err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH -A hmt03
#SBATCH -p h200

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Start time: $(date)"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Optional module loading, e.g.
#   sbatch --export=ALL,MODULES="cuda/12.1 gcc/11" scripts/train_gpu.sh
if [[ -n "${MODULES:-}" ]]; then
  for module_name in ${MODULES}; do
    module load "${module_name}"
  done
fi

# Optional conda environment, e.g.
#   sbatch --export=ALL,CONDA_ENV=pb-occ-alignn-full-tc scripts/train_gpu.sh
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "${HOME}/.bashrc"
  conda activate "${CONDA_ENV}"
fi

PYTHON_BIN="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"
DATA_CSV="${DATA_CSV:-data/toy/toy_data.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/real_exp}"
RESUME="${RESUME:-}"

echo "Python executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "Config: ${CONFIG}"
echo "Data CSV: ${DATA_CSV}"
echo "Output dir: ${OUTPUT_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. This is normal on some login nodes, but unexpected inside many GPU jobs."
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${DATA_CSV}" ]]; then
  echo "ERROR: data CSV not found: ${DATA_CSV}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print("ERROR: PyTorch is not installed in the active environment.", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(2)

print("torch:", torch.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available to PyTorch in this job.", file=sys.stderr)
    print("Check that this job is running on a GPU node and that GPU-enabled PyTorch is installed.", file=sys.stderr)
    sys.exit(3)

print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ -n "${RESUME}" ]]; then
  "${PYTHON_BIN}" scripts/train.py \
    --config "${CONFIG}" \
    --data_csv "${DATA_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --resume "${RESUME}" \
    --device cuda
else
  "${PYTHON_BIN}" scripts/train.py \
    --config "${CONFIG}" \
    --data_csv "${DATA_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda
fi

echo "End time: $(date)"
echo "Done."
