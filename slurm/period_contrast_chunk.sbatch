#!/bin/bash
# Parametrized decade-chunk sbatch for period_contrast. The io QoS caps each
# job at 4 CPUs and ~21 GB memory, so the heavier periods (Pre-sat 40 yr,
# Modern 35 yr) cannot materialise in a single job. Each invocation owns one
# decade-sized sub-window; the cached period-internal climatology keeps the
# GW detection threshold consistent across chunks.
#
# Submit with:
#   sbatch --job-name=arpc_modern_1990s \
#     --export=ALL,PERIOD_NAME=modern_1980_1999,\
# OUT_SUFFIX=modern_1990s,TIME_RANGE=1990-01-01:1999-12-31 \
#     slurm/period_contrast_chunk.sbatch
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

: "${PERIOD_NAME:?PERIOD_NAME env var required}"
: "${OUT_SUFFIX:?OUT_SUFFIX env var required}"
: "${TIME_RANGE:?TIME_RANGE env var required (YYYY-MM-DD:YYYY-MM-DD)}"

python -m src.analysis.period_contrast \
    --periods "$PERIOD_NAME" \
    --out-suffix "$OUT_SUFFIX" \
    --time-range "$TIME_RANGE"
