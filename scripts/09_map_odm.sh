#!/usr/bin/env bash
# Orthophoto + DSM + point cloud with OpenDroneMap (Docker, GPU feature extraction).
# perspective lens: principal point fixed at the centre; with nadir-only flat scenes the brown
# model drifts it far off-centre and dense matching fails.
# -w /tmp: PoissonRecon writes temp files to cwd, which must be writable by the host user.
# Input: data/map/odm/rail from 08_map_prepare.py. Output: data/map/odm/rail/odm_*.
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS="$PWD/data/map/odm"
IMAGE=${ODM_IMAGE:-opendronemap/odm:gpu}

docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --user "$(id -u):$(id -g)" -w /tmp \
  -v "$DATASETS:/datasets" "$IMAGE" \
  --project-path /datasets rail \
  --geo /datasets/rail/geo.txt \
  --camera-lens perspective \
  --feature-quality high \
  --min-num-features 12000 \
  --matcher-neighbors 24 \
  --pc-quality high \
  --dsm \
  --dem-resolution 1.0 \
  --orthophoto-resolution 0.5 \
  --max-concurrency 8 \
  "$@"
