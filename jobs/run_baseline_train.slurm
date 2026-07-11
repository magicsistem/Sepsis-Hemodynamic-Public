#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --time=24:00:00
#SBATCH --job-name=sepsis_base_train
#SBATCH --output=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic/logs/slurm_out_%j.txt
#SBATCH --error=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic/logs/slurm_err_%j.txt

set -euo pipefail

PROJECT_DIR=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic
SIF_IMAGE=/home/miguel.benavides__yachaytech.edu.ec/pytorch_24.01-py3.sif
DATA_DIR=$PROJECT_DIR/data/processed
OUTPUT_DIR=$PROJECT_DIR/results/baseline
CACHE_FILE=$OUTPUT_DIR/precomputed_features_baseline_v2.joblib
TRAIN_SCRIPT=$PROJECT_DIR/src/training/train_zabihi_cudf.py

export OMP_NUM_THREADS=1
export PYTHONPATH="$PROJECT_DIR"
export PYTHONUSERBASE=/home/miguel.benavides__yachaytech.edu.ec/.local
export PATH="$PYTHONUSERBASE/bin:$PATH"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

printf '[%s] Starting baseline final train job\n' "$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual_or_unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "ERROR: SIF image not found: $SIF_IMAGE" >&2
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data directory not found: $DATA_DIR" >&2
    exit 1
fi

if [[ ! -f "$CACHE_FILE" ]]; then
    echo "ERROR: baseline feature cache not found: $CACHE_FILE" >&2
    echo "Run sbatch jobs/run_baseline_features.slurm first." >&2
    exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

nvidia-smi

export PROJECT_DIR DATA_DIR OUTPUT_DIR

singularity exec --nv "$SIF_IMAGE" bash -lc '
set -euo pipefail
cd "$PROJECT_DIR"
echo "Container working directory: $PWD"
python --version
python -c "import cudf, cupy, xgboost, cuml; print(\"GPU stack import OK: cudf, cupy, xgboost, cuml\")"
python src/training/train_zabihi_cudf.py   --data_dir "$DATA_DIR"   --output_dir "$OUTPUT_DIR"   --n_jobs 64   --n_gpus 2   --stage train
'

printf '[%s] Finished baseline final train job\n' "$(date --iso-8601=seconds)"
