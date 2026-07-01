#!/usr/bin/env bash
# =============================================================================
# run_teleop.sh — keyboard/stdin action source for the ROS 2 control node
# =============================================================================
# Publishes 7-DoF actions onto /vla/action in the exact OpenVLA action format,
# so it is a drop-in stand-in for a trained policy. This node needs ROS 2 but
# NOT a GPU/sim session, so it runs straight under the venv python.
#
#   ./run_teleop.sh        # interactive: type actions, press Enter (type 'help')
#
# Start ./run_control.sh in another terminal first (or alongside); both must
# share the same ROS_DOMAIN_ID (42 on the reference machine).
#
# To drive the arm from a trained OpenVLA model instead, replace this command
# with your own node that publishes to /vla/action — nothing else changes.
# =============================================================================
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

[ -f "$VENV_DIR/bin/activate" ] || {
    echo "ERROR: venv not found at $VENV_DIR — run ./setup_venv.sh first." >&2
    exit 1
}

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
exec python "$REPO_DIR/vla_control/teleop_node.py" "$@"
