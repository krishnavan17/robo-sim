#!/usr/bin/env bash
# =============================================================================
# run_rviz.sh — visualise the collector's observation stream in RViz2
# =============================================================================
# Opens RViz2 pre-configured to show the live observation topics the collector
# publishes when run with --publish:
#
#   /vla/observation/pointcloud   sensor_msgs/PointCloud2  (frame: world)
#   /vla/observation/image        sensor_msgs/Image rgb8   (frame: observation_camera)
#   /vla/observation/depth        sensor_msgs/Image 32FC1  (frame: observation_camera, off by default)
#
# It also broadcasts a static TF  world -> observation_camera  matching the
# camera pose in vla_collect/config.py (CameraConfig.position / look_at), so the
# camera axes appear in the right place and RViz has a TF tree to resolve the
# world-frame cloud against.
#
# Usage — run the collector with --publish in one terminal:
#
#     ./run_collect.sh --publish --num-episodes 5
#
# and this in a SECOND terminal (same machine / same ROS_DOMAIN_ID):
#
#     ./run_rviz.sh
#
# RViz2 needs no GPU or the project venv — just a sourced ROS 2 Jazzy. This
# script sources /opt/ros/jazzy if ROS isn't already on the path.
# =============================================================================
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="$REPO_DIR/tools/rviz/vla_observation.rviz"

# --- ROS 2 environment -------------------------------------------------------
if ! command -v rviz2 >/dev/null 2>&1; then
    ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
    [ -f "$ROS_SETUP" ] || {
        echo "ERROR: rviz2 not found and no ROS setup at $ROS_SETUP." >&2
        echo "       Source your ROS 2 install, or set ROS_SETUP=/path/to/setup.bash." >&2
        exit 1
    }
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
fi

command -v rviz2 >/dev/null 2>&1 || {
    echo "ERROR: rviz2 still not on PATH after sourcing ROS. Is rviz2 installed?" >&2
    exit 1
}

# --- static TF: world -> observation_camera ---------------------------------
# Quaternion (xyzw) derived from CameraConfig.position=(1.70,0,1.20),
# look_at=(0.45,0,0.10) via the same look-at math as scene._look_at_quat. If you
# move the camera in config.py, recompute these (see tools/preview_camera.py).
CAM_XYZ=("1.70" "0.0" "1.20")
CAM_QUAT_XYZW=("-0.353048" "0.0" "0.935605" "0.0")

echo ">> Broadcasting static TF  world -> observation_camera"
ros2 run tf2_ros static_transform_publisher \
    --x "${CAM_XYZ[0]}" --y "${CAM_XYZ[1]}" --z "${CAM_XYZ[2]}" \
    --qx "${CAM_QUAT_XYZW[0]}" --qy "${CAM_QUAT_XYZW[1]}" \
    --qz "${CAM_QUAT_XYZW[2]}" --qw "${CAM_QUAT_XYZW[3]}" \
    --frame-id world --child-frame-id observation_camera &
TF_PID=$!

# Stop the TF publisher when RViz exits (or this script is interrupted).
cleanup() {
    kill "$TF_PID" 2>/dev/null || true
    wait "$TF_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- RViz2 -------------------------------------------------------------------
echo ">> Launching RViz2 with $RVIZ_CONFIG"
rviz2 -d "$RVIZ_CONFIG"
