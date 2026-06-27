#!/usr/bin/env bash
# =============================================================================
# launch.sh — activate the robo-sim venv and start NVIDIA Isaac Sim
# =============================================================================
# Two modes:
#
#   ./launch.sh                 Launch the Isaac Sim GUI application.
#   ./launch.sh script.py [..]  Run a standalone Isaac Sim Python script using
#                               the venv's python (so SimulationApp + your pip
#                               deps + ROS 2 are all importable). Extra args are
#                               forwarded to the script.
#
# The venv (.venv) is created by ./setup_venv.sh. Its activate hook already
# sources the ROS 2 and Isaac Sim env scripts (PYTHONPATH / LD_LIBRARY_PATH).
# For the standalone-python mode we additionally export the few vars that
# Isaac's own python.sh sets (CARB_APP_PATH, ISAAC_PATH, EXP_PATH, LD_PRELOAD).
# =============================================================================
# NOTE: no `set -u` — the ROS 2 / Isaac Sim setup scripts we source reference
# unbound vars (e.g. AMENT_TRACE_SETUP_FILES) and are not nounset-safe.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

# --- sanity checks -----------------------------------------------------------
[ -f "$VENV_DIR/bin/activate" ] || {
    echo "ERROR: venv not found at $VENV_DIR — run ./setup_venv.sh first." >&2
    exit 1
}
[ -d "$ISAAC_SIM_ROOT/kit" ] || {
    echo "ERROR: Isaac Sim not found at ISAAC_SIM_ROOT=$ISAAC_SIM_ROOT" >&2
    exit 1
}

# --- activate the venv (also sources ROS 2 + Isaac Sim env via our hook) -----
echo ">> Activating venv: $VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- Isaac Sim env vars (mirrors ~/python.sh) --------------------------------
export RESOURCE_NAME="IsaacSim"
export CARB_APP_PATH="$ISAAC_SIM_ROOT/kit"
export ISAAC_PATH="$ISAAC_SIM_ROOT"
export EXP_PATH="$ISAAC_SIM_ROOT/apps"

if [ "$#" -eq 0 ]; then
    # ---- GUI mode -----------------------------------------------------------
    echo ">> Starting Isaac Sim GUI..."
    exec "$ISAAC_SIM_ROOT/isaac-sim.sh"
else
    # ---- standalone Python mode --------------------------------------------
    # WAR for missing libcarb.so (same as python.sh).
    export LD_PRELOAD="${LD_PRELOAD:-}${LD_PRELOAD:+:}$ISAAC_SIM_ROOT/kit/libcarb.so"
    echo ">> Running '$1' with venv python ($(python --version 2>&1))..."
    exec python "$@"
fi
