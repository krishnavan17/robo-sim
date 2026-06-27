#!/usr/bin/env bash
# =============================================================================
# setup_venv.sh — create the robo-sim virtual environment
# =============================================================================
# Builds a Python 3.12 venv that can import BOTH:
#   * the pip packages in requirements.txt (numpy, opencv, ultralytics, ...)
#   * the already-installed NVIDIA Isaac Sim 6.0.0  (isaacsim, omni, carb, pxr)
#   * the already-installed ROS 2 Jazzy            (rclpy, cv_bridge, *_msgs)
#
# Isaac Sim and ROS 2 are NOT pip-installed. They are exposed to the venv via:
#   * a .pth file that adds Isaac Sim's site-packages to sys.path
#   * sourcing the Isaac / ROS env scripts from the venv's activate hook so the
#     required PYTHONPATH and LD_LIBRARY_PATH (native .so libs) are set.
#
# Usage:
#   ./setup_venv.sh            # creates ./.venv
#   source .venv/bin/activate  # then use it
# =============================================================================
set -euo pipefail

# --- locations (override via env if your installs live elsewhere) ------------
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME}"          # standalone Isaac Sim root
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
VENV_DIR="${VENV_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv}"

# --- sanity checks -----------------------------------------------------------
[ -f "$ISAAC_SIM_ROOT/setup_python_env.sh" ] || {
    echo "ERROR: Isaac Sim not found at ISAAC_SIM_ROOT=$ISAAC_SIM_ROOT" >&2
    echo "       (expected \$ISAAC_SIM_ROOT/setup_python_env.sh)" >&2
    exit 1
}
[ -f "$ROS_SETUP" ] || {
    echo "ERROR: ROS 2 setup not found at ROS_SETUP=$ROS_SETUP" >&2
    exit 1
}

# --- create venv (allow access to system/ROS site dirs not needed; we wire    #
#     them explicitly so use an isolated venv) -------------------------------
echo ">> Creating venv at $VENV_DIR"
python3.12 -m venv "$VENV_DIR"

# --- install pip dependencies ------------------------------------------------
echo ">> Installing requirements.txt"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

# --- expose Isaac Sim's site-packages to the venv via a .pth file ------------
SITE_PKGS="$("$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cat > "$SITE_PKGS/isaacsim.pth" <<EOF
$ISAAC_SIM_ROOT/kit/python/lib/python3.12/site-packages
$ISAAC_SIM_ROOT/exts/isaacsim.simulation_app
EOF
echo ">> Wrote $SITE_PKGS/isaacsim.pth"

# --- hook the Isaac + ROS env scripts into venv activation -------------------
# These set PYTHONPATH (isaacsim/omni/pxr/rclpy) and LD_LIBRARY_PATH (native
# .so files) which a .pth file alone cannot provide.
ACTIVATE="$VENV_DIR/bin/activate"
cat >> "$ACTIVATE" <<EOF

# ---- robo-sim: Isaac Sim + ROS 2 environment (added by setup_venv.sh) ----
source "$ROS_SETUP"
source "$ISAAC_SIM_ROOT/setup_python_env.sh"
[ -f "$ISAAC_SIM_ROOT/setup_ros_env.sh" ] && source "$ISAAC_SIM_ROOT/setup_ros_env.sh"
# --------------------------------------------------------------------------
EOF
echo ">> Patched $ACTIVATE to source Isaac Sim + ROS 2 env"

cat <<EOF

Done. Use it with:
    source "$VENV_DIR/bin/activate"
    python -c "import numpy, cv2, ultralytics, rclpy; print('pip + ROS ok')"
    # Isaac Sim modules (isaacsim/omni/pxr) require a GPU/display session.
EOF
