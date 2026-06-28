# =============================================================================
# vla_control/ros_interface.py — ROS 2 topic contract for VLA-driven control
# =============================================================================
# The single source of truth for how the action source (teleop today, a trained
# OpenVLA policy tomorrow) and the Isaac Sim control node talk to each other.
#
# Keeping the topic names and the (un)packing helpers in one Isaac-free module
# means both ends agree on the wire format byte-for-byte, and the helpers can be
# unit-tested without booting the simulator. `rclpy` / `*_msgs` are imported
# lazily inside the helpers so this module can also be imported by tooling that
# only wants the constants (e.g. the launch scripts or docs generation).
#
# The action layout is IDENTICAL to what vla_collect writes to disk, so a model
# fine-tuned on that dataset can publish straight onto /vla/action:
#
#     action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
#       * dx..dyaw : delta end-effector pose, world frame (metres / radians)
#       * gripper  : absolute command, 1.0 = closed/grasping, 0.0 = open
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # only for type hints; never imported at runtime here
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import Float32MultiArray

# --- topic names -------------------------------------------------------------
# Namespaced under /vla so the whole control graph is easy to spot in `ros2
# topic list` and won't collide with other nodes on the same ROS_DOMAIN_ID.
ACTION_TOPIC = "/vla/action"                 # Float32MultiArray, len 7  (source -> sim)
STATE_TOPIC = "/vla/observation/state"       # Float32MultiArray, len 7  (sim -> source)
IMAGE_TOPIC = "/vla/observation/image"       # sensor_msgs/Image rgb8    (sim -> source)
DEPTH_TOPIC = "/vla/observation/depth"       # sensor_msgs/Image 32FC1   (sim -> source, metres)
POINTCLOUD_TOPIC = "/vla/observation/pointcloud"  # sensor_msgs/PointCloud2 xyz (sim -> source)
INSTRUCTION_TOPIC = "/vla/instruction"       # std_msgs/String           (latched task text)
RESET_TOPIC = "/vla/reset"                   # std_msgs/String (colour) -> re-scatter + retask

# --- vector layout (must match vla_collect.recorder) -------------------------
ACTION_DIM = 7
STATE_DIM = 7
ACTION_LAYOUT = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
STATE_LAYOUT = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")

# Gripper convention, shared with vla_collect.config.ActionConfig.
GRIPPER_CLOSED = 1.0
GRIPPER_OPEN = 0.0
GRIPPER_THRESHOLD = 0.5   # >= this in a command means "close"

# QoS depth used by every publisher/subscriber in this package. Control is the
# latest-sample-wins kind of stream, so a shallow queue is correct.
QOS_DEPTH = 10


def pack_action(
    delta_pose: np.ndarray | list[float],
    gripper: float,
) -> "Float32MultiArray":
    """Build the /vla/action message from a 6-vector delta + gripper scalar.

    `delta_pose` is [dx, dy, dz, droll, dpitch, dyaw] (world frame). `gripper`
    is the absolute command (1=close, 0=open); it is stored verbatim so a model
    that emits soft values in [0,1] round-trips losslessly.
    """
    from std_msgs.msg import Float32MultiArray

    delta = np.asarray(delta_pose, dtype=np.float32).reshape(-1)
    if delta.shape[0] != 6:
        raise ValueError(f"delta_pose must have 6 elements, got {delta.shape[0]}")
    msg = Float32MultiArray()
    msg.data = [float(v) for v in delta] + [float(gripper)]
    return msg


def unpack_action(msg: "Float32MultiArray") -> tuple[np.ndarray, float]:
    """Inverse of `pack_action`: returns (delta_pose[6], gripper_scalar).

    Raises ValueError on a malformed message so the control node can log and
    skip a bad command rather than feed garbage to the motion solver.
    """
    data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
    if data.shape[0] != ACTION_DIM:
        raise ValueError(
            f"action message must have {ACTION_DIM} elements, got {data.shape[0]}"
        )
    return data[:6].copy(), float(data[6])


def pack_state(
    eef_pos: np.ndarray,
    eef_rpy: np.ndarray,
    gripper: float,
) -> "Float32MultiArray":
    """Build the /vla/observation/state message: [x,y,z, roll,pitch,yaw, grip].

    This mirrors vla_collect's `observation.state` exactly, so whatever consumes
    the live stream sees the same proprio layout it was trained against.
    """
    from std_msgs.msg import Float32MultiArray

    pos = np.asarray(eef_pos, dtype=np.float32).reshape(3)
    rpy = np.asarray(eef_rpy, dtype=np.float32).reshape(3)
    msg = Float32MultiArray()
    msg.data = [*pos.tolist(), *rpy.tolist(), float(gripper)]
    return msg


def unpack_state(msg: "Float32MultiArray") -> np.ndarray:
    """Return the [7] proprio state vector from a /vla/observation/state msg."""
    data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
    if data.shape[0] != STATE_DIM:
        raise ValueError(
            f"state message must have {STATE_DIM} elements, got {data.shape[0]}"
        )
    return data


def image_msg_type():
    """Return the sensor_msgs/Image *class* (for create_publisher/subscription)."""
    from sensor_msgs.msg import Image

    return Image


def image_to_msg(rgb: np.ndarray, frame_id: str = "observation_camera") -> "Image":
    """Convert an HxWx3 uint8 RGB array to a sensor_msgs/Image (encoding rgb8).

    We build the message by hand rather than depend on cv_bridge so the only
    runtime requirement is `sensor_msgs` (cv_bridge pulls in a heavier stack and
    is easy to get version-skewed against the system OpenCV).
    """
    from sensor_msgs.msg import Image

    rgb = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
    h, w = rgb.shape[:2]
    msg = Image()
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = rgb.tobytes()
    return msg


def msg_to_image(msg: "Image") -> np.ndarray:
    """Inverse of `image_to_msg`: rgb8 sensor_msgs/Image -> HxWx3 uint8 array."""
    if msg.encoding != "rgb8":
        raise ValueError(f"expected rgb8 image, got '{msg.encoding}'")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return buf.reshape(msg.height, msg.width, 3).copy()


# --- depth (sensor_msgs/Image, 32FC1) ----------------------------------------
def depth_to_msg(depth: np.ndarray, frame_id: str = "observation_camera") -> "Image":
    """Convert an HxW float32 metric-depth array to a sensor_msgs/Image (32FC1).

    Depth is the distance-to-image-plane in metres (NaN/inf for rays that hit
    nothing). We publish it as a single-channel float image — the standard ROS
    representation for a depth map — so RViz, depth_image_proc, and learning
    pipelines can consume it directly.
    """
    from sensor_msgs.msg import Image

    depth = np.ascontiguousarray(depth).astype(np.float32)
    h, w = depth.shape[:2]
    msg = Image()
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = w * 4                       # 4 bytes per float32 pixel, 1 channel
    msg.data = depth.tobytes()
    return msg


def msg_to_depth(msg: "Image") -> np.ndarray:
    """Inverse of `depth_to_msg`: 32FC1 sensor_msgs/Image -> HxW float32 array."""
    if msg.encoding != "32FC1":
        raise ValueError(f"expected 32FC1 depth image, got '{msg.encoding}'")
    buf = np.frombuffer(bytes(msg.data), dtype=np.float32)
    return buf.reshape(msg.height, msg.width).copy()


# --- pointcloud (sensor_msgs/PointCloud2, xyz float32) ------------------------
def pointcloud_msg_type():
    """Return the sensor_msgs/PointCloud2 *class* (for create_publisher)."""
    from sensor_msgs.msg import PointCloud2

    return PointCloud2


def pointcloud_to_msg(points: np.ndarray, frame_id: str = "world") -> "PointCloud2":
    """Convert an (N,3) float32 xyz array to an unorganised sensor_msgs/PointCloud2.

    The cloud is packed as three float32 fields (x, y, z), 12 bytes per point,
    which is the layout RViz and pcl_ros expect. `frame_id` should be the frame
    the points live in — "world" when scene.capture_pointcloud() is in world
    frame (the default), else the camera frame.
    """
    from sensor_msgs.msg import PointCloud2, PointField

    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.height = 1                         # unorganised cloud: 1 x N
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12                    # 3 * float32
    msg.row_step = msg.point_step * n
    msg.data = pts.tobytes()
    msg.is_dense = bool(np.isfinite(pts).all())
    return msg


def msg_to_pointcloud(msg: "PointCloud2") -> np.ndarray:
    """Inverse of `pointcloud_to_msg`: xyz PointCloud2 -> (N,3) float32 array.

    Assumes the x,y,z-float32 layout `pointcloud_to_msg` writes (the only layout
    this package publishes). For arbitrary PointCloud2s, use a full reader.
    """
    buf = np.frombuffer(bytes(msg.data), dtype=np.float32)
    n = msg.width * msg.height
    return buf.reshape(n, msg.point_step // 4)[:, :3].copy()
