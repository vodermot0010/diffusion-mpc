#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENABLE_MPC=true bash "$SCRIPT_DIR/sim_diffusion_planner_runner.sh" "$@"
