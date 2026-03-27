#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,5,6,7}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

###################################
# User Configuration Section
###################################
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/data/wyf/lgq/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/data/wyf/lgq/nuplan/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/wyf/lgq/nuplan/dataset/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/data/wyf/lgq/nuplan/exp}"

SPLIT="${SPLIT:-val14}"
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
BRANCH_NAME="${BRANCH_NAME:-diffusion_planner_release}"
RUN_PYTHON_PATH="${RUN_PYTHON_PATH:-/data/wyf/conda_envs/diffusion_planner/bin/python}"
ARGS_FILE="${ARGS_FILE:-/data/wyf/lgq/Diffusion-Planner/checkpoints/args.json}"
CKPT_FILE="${CKPT_FILE:-/data/wyf/lgq/Diffusion-Planner/checkpoints/model.pth}"
ENABLE_MPC="${ENABLE_MPC:-false}"
DRY_RUN="${DRY_RUN:-false}"
AUTO_TUNE_WORKERS="${AUTO_TUNE_WORKERS:-true}"
WORKER_THREADS="${WORKER_THREADS:-auto}"
WORKER_THREADS_MAX="${WORKER_THREADS_MAX:-}"
SYSTEM_MEMORY_RESERVE_MIB="${SYSTEM_MEMORY_RESERVE_MIB:-16384}"
PLANNER_OVERHEAD_MIB="${PLANNER_OVERHEAD_MIB:-}"
PER_SIM_MEMORY_MIB="${PER_SIM_MEMORY_MIB:-}"
GPUS_PER_SIM="${GPUS_PER_SIM:-0.05}"
###################################

if [[ ! -f "$ARGS_FILE" ]]; then
  echo "[ERROR] args file not found: $ARGS_FILE"
  exit 1
fi

if [[ ! -f "$CKPT_FILE" ]]; then
  echo "[ERROR] checkpoint not found: $CKPT_FILE"
  exit 1
fi

if [[ ! -x "$RUN_PYTHON_PATH" ]]; then
  echo "[ERROR] python executable not found: $RUN_PYTHON_PATH"
  exit 1
fi

if [[ "$SPLIT" == "val14" ]]; then
  SCENARIO_BUILDER="nuplan"
  SCENARIO_DATA_ROOT="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val"
else
  SCENARIO_BUILDER="nuplan_challenge"
  SCENARIO_DATA_ROOT="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val"
fi

if [[ "$ENABLE_MPC" == "true" ]]; then
  PLANNER="diffusion_planner_mpc"
  PLANNER_PREFIX="planner.diffusion_planner_mpc"
  RUN_TAG="mpc"
  DEFAULT_WORKER_THREADS_MAX=32
  DEFAULT_PLANNER_OVERHEAD_MIB=61440
  DEFAULT_PER_SIM_MEMORY_MIB=2560
else
  PLANNER="diffusion_planner"
  PLANNER_PREFIX="planner.diffusion_planner"
  RUN_TAG="base"
  DEFAULT_WORKER_THREADS_MAX=64
  DEFAULT_PLANNER_OVERHEAD_MIB=30720
  DEFAULT_PER_SIM_MEMORY_MIB=1280
fi

WORKER_THREADS_MAX="${WORKER_THREADS_MAX:-$DEFAULT_WORKER_THREADS_MAX}"
PLANNER_OVERHEAD_MIB="${PLANNER_OVERHEAD_MIB:-$DEFAULT_PLANNER_OVERHEAD_MIB}"
PER_SIM_MEMORY_MIB="${PER_SIM_MEMORY_MIB:-$DEFAULT_PER_SIM_MEMORY_MIB}"

detect_background_memory_mib() {
  ps -eo rss=,cmd= | awk '
    /data_process.py/ || /train_predictor.py/ || /torchrun/ || /python .*data_process.py/ {
      sum += $1
    }
    END {
      print int(sum / 1024)
    }
  '
}

auto_tune_worker_threads() {
  local cpu_count mem_available_mib background_memory_mib usable_memory_mib threads_by_mem threads
  cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)
  mem_available_mib=$(awk '/MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
  background_memory_mib=$(detect_background_memory_mib)
  usable_memory_mib=$((mem_available_mib - SYSTEM_MEMORY_RESERVE_MIB - PLANNER_OVERHEAD_MIB - background_memory_mib))

  if (( usable_memory_mib <= PER_SIM_MEMORY_MIB )); then
    threads=4
  else
    threads_by_mem=$((usable_memory_mib / PER_SIM_MEMORY_MIB))
    threads=$threads_by_mem
  fi

  if (( threads < 4 )); then
    threads=4
  fi
  if (( threads > cpu_count )); then
    threads=$cpu_count
  fi
  if (( threads > WORKER_THREADS_MAX )); then
    threads=$WORKER_THREADS_MAX
  fi

  echo "$threads"
}

if [[ "$AUTO_TUNE_WORKERS" == "true" ]]; then
  if [[ -z "${WORKER_THREADS}" || "${WORKER_THREADS}" == "auto" ]]; then
    WORKER_THREADS=$(auto_tune_worker_threads)
    AUTO_TUNED_WORKERS=true
  else
    AUTO_TUNED_WORKERS=false
  fi
else
  AUTO_TUNED_WORKERS=false
fi

FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)

CMD=(
  "$RUN_PYTHON_PATH" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py"
  "+simulation=$CHALLENGE"
  "planner=$PLANNER"
  "$PLANNER_PREFIX.config.args_file=$ARGS_FILE"
  "$PLANNER_PREFIX.ckpt_path=$CKPT_FILE"
  "scenario_builder=$SCENARIO_BUILDER"
  "scenario_filter=$SPLIT"
  "scenario_builder.data_root=$SCENARIO_DATA_ROOT"
  "experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_${RUN_TAG}_$TIMESTAMP"
  "verbose=true"
  "worker=ray_distributed"
  "worker.threads_per_node=$WORKER_THREADS"
  "distributed_mode=SINGLE_NODE"
  "number_of_gpus_allocated_per_simulation=$GPUS_PER_SIM"
  "enable_simulation_progress_bar=true"
  "hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
)

echo "Processing $CKPT_FILE"
echo "Planner: $PLANNER"
echo "Split: $SPLIT"
echo "Challenge: $CHALLENGE"
echo "Worker threads: $WORKER_THREADS"

if [[ "$AUTO_TUNED_WORKERS" == "true" ]]; then
  echo "Worker tuning: auto"
  echo "  MemAvailable(MiB): $(awk '/MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)"
  echo "  Background reserve(MiB): $(detect_background_memory_mib)"
  echo "  Planner overhead(MiB): $PLANNER_OVERHEAD_MIB"
  echo "  Per worker budget(MiB): $PER_SIM_MEMORY_MIB"
  echo "  Worker cap: $WORKER_THREADS_MAX"
fi

if [[ "$DRY_RUN" == "true" ]]; then
  printf "[DRY_RUN] "
  printf "%q " "${CMD[@]}"
  printf "\n"
  exit 0
fi

"${CMD[@]}"
