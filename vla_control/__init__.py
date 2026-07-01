"""vla_control — ROS 2 closed-loop control of the UR10 in Isaac Sim.

This package drives the *same* table-top pick-and-place world that
``vla_collect`` records (a 6-axis UR10 arm, four coloured cubes, a red target
pad, and a fixed third-person camera) but instead of replaying a scripted
RMPFlow demonstration it executes **7-DoF actions received over ROS 2** — the
exact action vector OpenVLA emits.

Architecture (three ROS 2 participants)::

      ┌──────────────┐  /vla/action (Float32MultiArray[7])   ┌────────────────┐
      │ action source │ ─────────────────────────────────────▶│ control_node   │
      │ (teleop now,  │                                        │ (Isaac Sim)    │
      │  VLA later)   │ ◀───────────────────────────────────  │  UR10 + RMPFlow │
      └──────────────┘   /vla/observation/{image,state,        └────────────────┘
                                            depth,pointcloud}
                         /vla/instruction (String)

The action vector is identical to the one ``vla_collect`` writes to disk:

    action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
      * dx..dyaw : delta end-effector pose (world frame, metres / radians)
      * gripper  : absolute, 1.0 = closed/grasping, 0.0 = open

Because the control node consumes that vector verbatim, a trained OpenVLA model
can be dropped in later with no change to the simulator side: replace the
teleop node with a node that publishes the model's predicted action to the same
``/vla/action`` topic.

Submodules:
    ros_interface — topic names + (un)packing helpers. No Isaac/ROS *runtime*.
    control_node  — standalone Isaac Sim entry point (the simulator side).
    teleop_node   — pure-ROS node that reads actions from stdin (the source side).
"""

__all__ = ["ros_interface"]
