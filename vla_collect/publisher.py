# =============================================================================
# vla_collect/publisher.py — optional ROS 2 mirror of the recorded observations
# =============================================================================
# While the collector drives the scripted RMPFlow rollout it ALREADY produces
# exactly the observations vla_control streams live: a 224x224 RGB frame, the
# 7-DoF proprio state, (optionally) metric depth + a pointcloud, and the task
# instruction. This node re-publishes each recorded frame onto the SAME ROS 2
# topics the vla_control node serves, so a live consumer (RViz, a logger, a
# policy being evaluated against the scripted demos) sees an identical stream
# whether the data comes from the collector or from closed-loop control.
#
# It reuses vla_control.ros_interface verbatim — the one source of truth for the
# topic names and the message (un)packing — so the wire format is byte-for-byte
# what control publishes. This is publish-only: the collector generates its own
# actions from the scripted policy, so there is nothing to subscribe to here.
#
# rclpy is imported lazily inside __init__ so `import vla_collect.publisher`
# costs nothing (and needs no ROS env) unless a run actually enables publishing.
# =============================================================================
from __future__ import annotations

import numpy as np

from vla_control import ros_interface as ri
from vla_collect.recorder import quat_to_rpy


class ObservationPublisher:
    """ROS 2 node that mirrors recorded frames onto the vla_control topics.

    Usage (only when --publish is set; see collect.py)::

        import rclpy
        rclpy.init()
        pub = ObservationPublisher(cfg)
        pub.publish_instruction("place the blue cube on the red rectangle")
        # each recorded frame:
        pub.publish_observation(image, eef_pos, eef_quat_wxyz, gripper_closed,
                                depth=depth, pointcloud=cloud)
        ...
        pub.destroy()
    """

    def __init__(self, cfg) -> None:
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray, String

        self.cfg = cfg
        # A plain Node (not a subclass) so this module imports without rclpy.
        self.node = Node("vla_collect_publisher")

        self.pub_state = self.node.create_publisher(
            Float32MultiArray, ri.STATE_TOPIC, ri.QOS_DEPTH
        )
        self.pub_image = self.node.create_publisher(
            ri.image_msg_type(), ri.IMAGE_TOPIC, ri.QOS_DEPTH
        )
        # Depth (32FC1) and pointcloud (PointCloud2) only when the camera is
        # configured to produce them, exactly as the control node decides.
        self.pub_depth = (
            self.node.create_publisher(ri.image_msg_type(), ri.DEPTH_TOPIC, ri.QOS_DEPTH)
            if cfg.camera.enable_depth
            else None
        )
        self.pub_pointcloud = (
            self.node.create_publisher(
                ri.pointcloud_msg_type(), ri.POINTCLOUD_TOPIC, ri.QOS_DEPTH
            )
            if cfg.camera.enable_pointcloud
            else None
        )
        # Instruction is latched (transient-local) so a consumer that subscribes
        # mid-run still receives the current task — same QoS as vla_control.
        self.pub_instruction = self.node.create_publisher(
            String, ri.INSTRUCTION_TOPIC, _latched_qos()
        )

    # --- publishing ----------------------------------------------------------
    def publish_observation(
        self,
        image: np.ndarray,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
        gripper_closed: bool,
        depth: np.ndarray | None = None,
        pointcloud: np.ndarray | None = None,
    ) -> None:
        """Publish one recorded frame on the observation topics.

        Mirrors VlaControlNode._publish_observation: RGB + proprio state always,
        depth/pointcloud only when enabled and a valid frame is available (the
        annotators can return None until the renderer has warmed up).
        """
        self.pub_image.publish(ri.image_to_msg(image))

        if self.pub_depth is not None and depth is not None:
            self.pub_depth.publish(ri.depth_to_msg(depth))
        if self.pub_pointcloud is not None and pointcloud is not None:
            frame = "world" if self.cfg.camera.pointcloud_world_frame else "observation_camera"
            self.pub_pointcloud.publish(ri.pointcloud_to_msg(pointcloud, frame_id=frame))

        rpy = quat_to_rpy(np.asarray(eef_quat_wxyz, dtype=np.float64).reshape(4))
        gripper = ri.GRIPPER_CLOSED if gripper_closed else ri.GRIPPER_OPEN
        self.pub_state.publish(ri.pack_state(eef_pos, rpy, gripper))

    def publish_instruction(self, instruction: str) -> None:
        """Publish the (latched) task instruction, as vla_control does on reset."""
        from std_msgs.msg import String

        msg = String()
        msg.data = instruction
        self.pub_instruction.publish(msg)

    def destroy(self) -> None:
        self.node.destroy_node()


def _latched_qos():
    """Transient-local QoS so late subscribers still get the latest instruction.

    Mirrors vla_control.controller._latched_qos so the instruction topic behaves
    identically whether served by the collector or the control node.
    """
    from rclpy.qos import DurabilityPolicy, QoSProfile

    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
