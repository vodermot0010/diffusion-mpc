#!/bin/bash
set -ex

RUN_PYTHON_PATH="/data/wyf/conda_envs/diffusion_planner/bin/python"
NUPLAN_DATA_PATH="/data/wyf/lgq/nuplan/dataset/nuplan-v1.1/splits/train"
NUPLAN_MAP_PATH="/data/wyf/lgq/nuplan/dataset/maps"
SAVE_PATH="/data/wyf/lgq/nuplan/dataset/nuplan-v1.1/splits/train_processed"

mkdir -p "$SAVE_PATH"

$RUN_PYTHON_PATH -u data_process.py \
  --data_path "$NUPLAN_DATA_PATH" \
  --map_path "$NUPLAN_MAP_PATH" \
  --save_path "$SAVE_PATH" \
  --total_scenarios 100000 \
  --refresh_db_validation_cache True