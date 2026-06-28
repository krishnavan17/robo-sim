#!/usr/bin/env bash
# =============================================================================
# run_control.sh — ROS 2 closed-loop control of the UR10 pick-place world
# =============================================================================
# Thin wrapper over ./launch.sh that runs the standalone Isaac Sim control node
# under the venv + Isaac Sim + ROS 2. The node subscribes to /vla/action and
# drives the arm; pair it with an action source (run_teleop.sh today, a trained
# OpenVLA policy later). All arguments are forwarded to control_node.py:
#
#   ./run_control.sh                       # task "place the blue cube ..."
#   ./run_control.sh --color green         # different cube colour
#   ./run_control.sh --gui                  # watch it in the viewport
#   ./run_control.sh --seed 1               # different cube layout
#
# Run the action source in a SECOND terminal (same machine / ROS_DOMAIN_ID):
#   ./run_teleop.sh
# =============================================================================
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$REPO_DIR/launch.sh" "$REPO_DIR/vla_control/control_node.py" "$@"
