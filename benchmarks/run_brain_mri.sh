#!/bin/bash
# Validate the Brain Tumor MRI and Alzheimer MRI datasets with SimCLR and a supervised baseline (ResNet-18).
#
# Setup, from the repo root. The benchmarks need the stable-pretraining version from before simclr_forward was renamed:
#   pip install -e ".[benchmarks]"
#   pip install "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining@4ba07974de70fc3cd171838890603a3ea6e62be7"
#
# Usage, from the repo root:
#   bash benchmarks/run_brain_mri.sh          # full runs
#   bash benchmarks/run_brain_mri.sh smoke    # 3 batches per run to check the setup
set -euo pipefail

# Store datasets, checkpoints and logs here instead of the cluster paths in config.yaml.
export STABLE_DATASETS_ROOT="${STABLE_DATASETS_ROOT:-$HOME/brain_mri_runs}"
export PROJECT_ROOT="${PROJECT_ROOT:-$STABLE_DATASETS_ROOT}"

SMOKE_TEST=false
[ "${1:-}" = "smoke" ] && SMOKE_TEST=true

for dataset in braintumormri alzheimermri; do
    for model in simclr supervised; do
        echo "=== $model on $dataset ==="
        python -m benchmarks.run \
            model="$model" \
            backbone=resnet18 \
            dataset="$dataset" \
            seed=0 \
            smoke_test="$SMOKE_TEST" \
            wandb.enabled="${WANDB_ENABLED:-false}"
    done
done
