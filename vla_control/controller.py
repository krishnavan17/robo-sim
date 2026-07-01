# =============================================================================
# vla_control/controller.py — the ROS 2 node + RMPFlow servo loop (Isaac side)
# =============================================================================
# Imported ONLY after SimulationApp() exists (it pulls in isaacsim.* and the
# scene). Owns the rclpy node, the simulated world, and the control loop that
# turns incoming 7-DoF delta actions into UR10 motion.
#
# How an action becomes motion
# ----------------------------
# OpenVLA (and our teleop) emit *delta* end-effector actions, exactly as stored
# by vla_collect:  [dx, dy, dz, droll, dpitch, dyaw, gripper].
#
#   1. We keep a running TARGET end-effector pose (position + rpy), seeded from
#      the arm's actual pose at reset.
#   2. Each received action adds its (clipped) delta onto that target and sets
#      the absolute gripper command. The target position is clamped to a safe
#      workspace box so a runaway delta stream can't drive the solver off-table.
#   3. RMPFlowController servos the arm toward the target over the next few
#      physics steps (one action == one ~20 Hz dataset step), and the surface
#      gripper is opened/closed to match the command.
#   4. After stepping, we publish the fresh camera image + proprio state so the
#      action source can compute its next action — closing the loop.
#
# Integrating deltas (rather than treating the action as an absolute goal) is
# what makes this compatible with a delta-trained policy AND with hand-entered
# teleop nudges: both just say "move the hand this much from where it is now".
# =============================================================================
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.universal_robots.controllers import (
    RMPFlowController,
)

from vla_collect.recorder import quat_to_rpy, wrap_to_pi
from vla_collect.scene import PickPlaceScene
from vla_control import ros_interface as ri
from vla_control.config import CUBES, ControlConfig


class VlaControlNode(Node):
    """ROS 2 node that servos the UR10 from /vla/action and streams observations."""

    def __init__(self, cfg: ControlConfig, sim_app) -> None:
        super().__init__("vla_control_node")
        self.cfg = cfg
        self.sim_app = sim_app
        self.rng = np.random.default_rng(cfg.seed)

        self.scene = PickPlaceScene(_as_collect_cfg(cfg))
        self.rmp: RMPFlowController | None = None

        # Running control target (set in build_and_reset once the arm exists).
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._target_rpy = np.zeros(3, dtype=np.float64)
        self._gripper_cmd = ri.GRIPPER_OPEN

        # The most recent action awaiting application, and a flag so we only
        # advance the target when a genuinely new command has arrived.
        self._pending_action: tuple[np.ndarray, float] | None = None
        self.color = cfg.initial_color

        # --- ROS wiring ------------------------------------------------------
        self.sub_action = self.create_subscription(
            Float32MultiArray, ri.ACTION_TOPIC, self._on_action, ri.QOS_DEPTH
        )
        self.sub_reset = self.create_subscription(
            String, ri.RESET_TOPIC, self._on_reset, ri.QOS_DEPTH
        )
        self.pub_state = self.create_publisher(
            Float32MultiArray, ri.STATE_TOPIC, ri.QOS_DEPTH
        )
        self.pub_image = self.create_publisher(
            ri.image_msg_type(), ri.IMAGE_TOPIC, ri.QOS_DEPTH
        )
        # Depth (32FC1) and pointcloud (PointCloud2) observations for training
        # depth-/geometry-conditioned models. Only created when the camera is
        # configured to produce them, so an RGB-only run pays nothing.
        self.pub_depth = (
            self.create_publisher(ri.image_msg_type(), ri.DEPTH_TOPIC, ri.QOS_DEPTH)
            if cfg.camera.enable_depth
            else None
        )
        self.pub_pointcloud = (
            self.create_publisher(ri.pointcloud_msg_type(), ri.POINTCLOUD_TOPIC, ri.QOS_DEPTH)
            if cfg.camera.enable_pointcloud
            else None
        )
        # Instruction is latched (transient-local) so a source that subscribes
        # late still receives the current task. Built explicitly for that QoS.
        self.pub_instruction = self.create_publisher(
            String, ri.INSTRUCTION_TOPIC, _latched_qos()
        )

    # --- lifecycle -----------------------------------------------------------
    def build_and_reset(self) -> None:
        """Construct the world, reset physics, seed the control target."""
        print(">> Building scene")
        self.scene.build()
        self.scene.reset()                 # initialises robot, gripper, camera
        # RMPFlow drives the arm toward EE targets; the suction variant matches
        # the gripper attached in PickPlaceScene.
        self.rmp = RMPFlowController(
            name="vla_rmpflow",
            robot_articulation=self.scene.robot,
            attach_gripper=True,
        )
        self._reset_episode(self.color)
        self._publish_instruction()
        print(f">> Ready. Task: '{self.cfg.instruction_for(self.color)}'")
        print(f">> Listening on {ri.ACTION_TOPIC} (7-DoF: {', '.join(ri.ACTION_LAYOUT)})")

    def _reset_episode(self, color: str) -> None:
        """Re-scatter the cubes and re-seed the running target from the arm."""
        self.color = color
        self.scene.world.reset()
        self.scene.randomise_cubes(self.rng)
        self.rmp.reset()
        self.scene.robot.gripper.open()
        self._gripper_cmd = ri.GRIPPER_OPEN
        # Let physics settle and the camera warm up before we read the pose.
        for _ in range(int(0.3 / self.cfg.physics_dt)):
            self.scene.world.step(render=True)
        # Seed the control target at the arm's CURRENT end-effector pose, so the
        # first delta moves relative to where the hand actually is.
        pos, quat = self.scene.end_effector_pose()
        self._target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        self._target_rpy = quat_to_rpy(np.asarray(quat, dtype=np.float64).reshape(4))
        self._pending_action = None

    # --- ROS callbacks -------------------------------------------------------
    def _on_action(self, msg: Float32MultiArray) -> None:
        try:
            delta, gripper = ri.unpack_action(msg)
        except ValueError as exc:
            self.get_logger().warn(f"ignoring malformed action: {exc}")
            return
        self._pending_action = (delta, gripper)

    def _on_reset(self, msg: String) -> None:
        color = (msg.data or "").strip().lower()
        if color not in {c.name for c in CUBES}:
            self.get_logger().warn(
                f"reset colour '{msg.data}' not in {[c.name for c in CUBES]}; "
                f"keeping '{self.color}'"
            )
            color = self.color
        self.get_logger().info(f"reset -> task colour '{color}'")
        self._reset_episode(color)
        self._publish_instruction()

    # --- target integration --------------------------------------------------
    def _integrate_action(self, delta: np.ndarray, gripper: float) -> None:
        """Add a clipped delta onto the running target; clamp to the safe box."""
        ac = self.cfg.action
        dpos = np.clip(delta[:3], -ac.max_delta_pos, ac.max_delta_pos)
        drot = np.clip(delta[3:6], -ac.max_delta_rot, ac.max_delta_rot)

        self._target_pos = self._target_pos + dpos
        self._target_pos[0] = np.clip(self._target_pos[0], *self.cfg.target_x)
        self._target_pos[1] = np.clip(self._target_pos[1], *self.cfg.target_y)
        self._target_pos[2] = np.clip(self._target_pos[2], *self.cfg.target_z)
        # Keep orientation wrapped to (-pi, pi] so it never winds up unbounded.
        self._target_rpy = wrap_to_pi(self._target_rpy + drot)
        self._gripper_cmd = float(gripper)

    def _apply_gripper(self) -> None:
        """Drive the suction gripper toward the absolute command."""
        want_closed = self._gripper_cmd >= ri.GRIPPER_THRESHOLD
        is_closed = self.scene.robot.gripper.is_closed()
        if want_closed and not is_closed:
            self.scene.robot.gripper.close()
        elif not want_closed and is_closed:
            self.scene.robot.gripper.open()

    # --- the loop ------------------------------------------------------------
    def run(self) -> None:
        """Spin ROS + physics together until the app or ROS shuts down.

        One iteration == one received action held for `physics_steps_per_action`
        physics ticks (so a 60 Hz sim presents ~20 Hz control, matching the
        dataset). Observations are published every `publish_every_n_physics_steps`.
        """
        cfg = self.cfg
        phys_step = 0
        while self.sim_app.is_running() and rclpy.ok():
            # Drain any queued ROS callbacks (new action / reset) without blocking.
            rclpy.spin_once(self, timeout_sec=0.0)

            # Advance the target only when a fresh action has arrived; otherwise
            # keep servoing toward the last target (finish the in-flight motion).
            if self._pending_action is not None:
                delta, gripper = self._pending_action
                self._integrate_action(delta, gripper)
                self._pending_action = None
            elif not cfg.hold_last_target:
                # Optionally idle in place by re-seeding the target to "here".
                pos, quat = self.scene.end_effector_pose()
                self._target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
                self._target_rpy = quat_to_rpy(np.asarray(quat).reshape(4))

            target_quat = euler_angles_to_quat(self._target_rpy)
            self._apply_gripper()

            # Hold this target for a few physics ticks so RMPFlow can move toward
            # it — one action maps to one dataset-rate control step.
            for _ in range(cfg.physics_steps_per_action):
                action = self.rmp.forward(
                    target_end_effector_position=self._target_pos,
                    target_end_effector_orientation=target_quat,
                )
                self.scene.robot.apply_action(action)
                self.scene.world.step(render=True)
                self.scene.robot.gripper.update()
                phys_step += 1
                if phys_step % cfg.publish_every_n_physics_steps == 0:
                    self._publish_observation()

    # --- publishing ----------------------------------------------------------
    def _publish_observation(self) -> None:
        image = self.scene.capture_image()
        self.pub_image.publish(ri.image_to_msg(image))
        # Depth + pointcloud, when the camera produces them. Both can be None for
        # the first few frames after a reset (annotator not yet warmed up); skip
        # publishing until valid data is available rather than send empties.
        if self.pub_depth is not None:
            depth = self.scene.capture_depth()
            if depth is not None:
                self.pub_depth.publish(ri.depth_to_msg(depth))
        if self.pub_pointcloud is not None:
            cloud = self.scene.capture_pointcloud()
            if cloud is not None:
                frame = "world" if self.cfg.camera.pointcloud_world_frame else "observation_camera"
                self.pub_pointcloud.publish(ri.pointcloud_to_msg(cloud, frame_id=frame))
        pos, quat = self.scene.end_effector_pose()
        rpy = quat_to_rpy(np.asarray(quat, dtype=np.float64).reshape(4))
        gripper = ri.GRIPPER_CLOSED if self.scene.robot.gripper.is_closed() else ri.GRIPPER_OPEN
        self.pub_state.publish(ri.pack_state(pos, rpy, gripper))

    def _publish_instruction(self) -> None:
        msg = String()
        msg.data = self.cfg.instruction_for(self.color)
        self.pub_instruction.publish(msg)


def _as_collect_cfg(cfg: ControlConfig):
    """Adapt a ControlConfig into the CollectConfig shape PickPlaceScene reads.

    PickPlaceScene only touches `.workspace`, `.camera`, `.physics_dt`, and
    `.rendering_dt` — all present on ControlConfig with identical names — plus
    `.action.ee_offset`. We pass the ControlConfig straight through; it is a
    structural superset, so no copy is needed.
    """
    return cfg


def _latched_qos():
    """Transient-local QoS so late subscribers still get the latest instruction."""
    from rclpy.qos import DurabilityPolicy, QoSProfile

    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
