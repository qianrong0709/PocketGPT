#!/bin/bash
#SBATCH --job-name=PocketGPT
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Supply partition, GPU, CPU, memory and time limits explicitly to sbatch.
# Required exported variables: POCKETGPT_PROJECT and POCKETGPT_PYTHON.
# Usage: sbatch [resources] scripts/train.slurm /absolute/recipe.json
#        [--resume /absolute/checkpoint.pt] [--max-steps N]
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit this wrapper with sbatch; it must not train on a login node.' >&2
    exit 2
fi
if [[ $# -lt 1 || "$1" != /* ]]; then
    echo 'Usage: train.slurm /absolute/recipe.json [--resume /absolute/checkpoint.pt] [--max-steps N]' >&2
    exit 2
fi
if [[ "${POCKETGPT_PROJECT:-}" != /* || ! -d "${POCKETGPT_PROJECT:-}" ||
      "${POCKETGPT_PYTHON:-}" != /* || ! -x "${POCKETGPT_PYTHON:-}" ]]; then
    echo 'Export an existing absolute POCKETGPT_PROJECT and executable absolute POCKETGPT_PYTHON.' >&2
    exit 2
fi
pocketgpt_recipe="$1"
shift
if [[ ! -r "$pocketgpt_recipe" ||
      ! -f "$POCKETGPT_PROJECT/pocketgpt/__main__.py" ||
      ! -f "$POCKETGPT_PROJECT/scripts/server_preflight.py" ||
      ! -d "$POCKETGPT_PROJECT/tests" ]]; then
    echo 'The explicit recipe or project files are unavailable; refusing to fall back.' >&2
    exit 2
fi
pocketgpt_preflight_args=()
pocketgpt_train_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            if [[ $# -lt 2 || "$2" != /* || ${#pocketgpt_preflight_args[@]} -gt 0 ]]; then
                echo '--resume requires one absolute checkpoint path, supplied once.' >&2
                exit 2
            fi
            pocketgpt_preflight_args+=(--resume "$2")
            pocketgpt_train_args+=(--resume "$2")
            shift 2
            ;;
        --max-steps)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo '--max-steps requires a positive integer.' >&2
                exit 2
            fi
            pocketgpt_train_args+=(--max-steps "$2")
            shift 2
            ;;
        *) echo "Unexpected training option: $1" >&2; exit 2 ;;
    esac
done

unset PYTHONPATH PYTHONHOME BASH_ENV ENV HTMD_CONFIG
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=disabled WANDB_DISABLED=true DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$POCKETGPT_PROJECT"

echo "PocketGPT Slurm job ${SLURM_JOB_ID}: checking the explicit local recipe before training."
"$POCKETGPT_PYTHON" scripts/server_preflight.py \
    --config "$pocketgpt_recipe" --device cuda \
    --project-root "$POCKETGPT_PROJECT" ${pocketgpt_preflight_args[@]+"${pocketgpt_preflight_args[@]}"}
"$POCKETGPT_PYTHON" -m unittest discover -s tests -v
exec "$POCKETGPT_PYTHON" -m pocketgpt train \
    --config "$pocketgpt_recipe" --device cuda ${pocketgpt_train_args[@]+"${pocketgpt_train_args[@]}"}
