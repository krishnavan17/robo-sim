# =============================================================================
# vla_control/teleop_node.py — keyboard/stdin action source (stand-in for OpenVLA)
# =============================================================================
# Publishes 7-DoF actions onto /vla/action in the SAME format a trained OpenVLA
# policy emits, so the simulator side (control_node.py) can't tell the two apart.
# This is the "for this implementation, accept user input in the same format"
# piece: today a human types actions; tomorrow swap this node for one that
# publishes model.predict(image, instruction) to the identical topic.
#
# This node imports ONLY ROS 2 + numpy (no Isaac Sim), so it runs under the venv
# without a GPU/sim session:
#
#     ./run_teleop.sh                 # interactive prompt
#     ./launch.sh vla_control/teleop_node.py
#
# Two input modes on stdin:
#
#   1. Raw action vector — 7 numbers, the exact OpenVLA action:
#          dx dy dz droll dpitch dyaw gripper
#      e.g.  0.0 0.0 -0.02 0 0 0 0      (descend 2 cm, gripper open)
#            0 0 0 0 0 0 1              (close gripper, no motion)
#
#   2. Mnemonic nudges (convenience, expanded to the 7-vector for you):
#          w/s  +x/-x      a/d  +y/-y      r/f  +z/-z
#          grip / open     reset <color>   q/quit
#      a bare Enter repeats a zero-motion action (holds pose, keeps streaming).
#
# Either way the published message is a plain Float32MultiArray of length 7 —
# nothing about mode 2 leaks onto the wire.
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_control import ros_interface as ri  # noqa: E402

# Step size for the mnemonic nudges (metres / radians). Kept under the action
# clip bounds so a single keypress is always applied in full by the sim.
_STEP_POS = 0.02
_STEP_ROT = 0.05

_NUDGES = {
    "w": ([+_STEP_POS, 0, 0, 0, 0, 0], None),
    "s": ([-_STEP_POS, 0, 0, 0, 0, 0], None),
    "a": ([0, +_STEP_POS, 0, 0, 0, 0], None),
    "d": ([0, -_STEP_POS, 0, 0, 0, 0], None),
    "r": ([0, 0, +_STEP_POS, 0, 0, 0], None),
    "f": ([0, 0, -_STEP_POS, 0, 0, 0], None),
    "roll+": ([0, 0, 0, +_STEP_ROT, 0, 0], None),
    "roll-": ([0, 0, 0, -_STEP_ROT, 0, 0], None),
    "pitch+": ([0, 0, 0, 0, +_STEP_ROT, 0], None),
    "pitch-": ([0, 0, 0, 0, -_STEP_ROT, 0], None),
    "yaw+": ([0, 0, 0, 0, 0, +_STEP_ROT], None),
    "yaw-": ([0, 0, 0, 0, 0, -_STEP_ROT], None),
    "grip": ([0, 0, 0, 0, 0, 0], ri.GRIPPER_CLOSED),
    "open": ([0, 0, 0, 0, 0, 0], ri.GRIPPER_OPEN),
}

_HELP = """\
Type an action and press Enter. Two accepted forms:

  • 7 numbers (the raw OpenVLA action vector):
        dx dy dz droll dpitch dyaw gripper
        e.g.  0 0 -0.02 0 0 0 0      0 0 0 0 0 0 1

  • a mnemonic nudge:
        w/s = +x/-x   a/d = +y/-y   r/f = +z/-z
        roll±  pitch±  yaw±          grip = close   open = release
        (bare Enter repeats a zero-motion hold)

  reset <color>   re-scatter cubes and retask  (blue/green/yellow/red)
  help            show this message
  q / quit        exit
"""


class TeleopNode(Node):
    """Reads actions from stdin and publishes them on /vla/action."""

    def __init__(self) -> None:
        super().__init__("vla_teleop_node")
        self.pub_action = self.create_publisher(
            Float32MultiArray, ri.ACTION_TOPIC, ri.QOS_DEPTH
        )
        self.pub_reset = self.create_publisher(
            String, ri.RESET_TOPIC, ri.QOS_DEPTH
        )
        # Latch the gripper across nudges so "grip" then "w" keeps it closed,
        # matching how the absolute gripper command behaves in the dataset.
        self._gripper = ri.GRIPPER_OPEN

    def publish_action(self, delta: list[float], gripper: float) -> None:
        self._gripper = gripper
        self.pub_action.publish(ri.pack_action(delta, gripper))

    def publish_reset(self, color: str) -> None:
        msg = String()
        msg.data = color
        self.pub_reset.publish(msg)

    # --- parsing -------------------------------------------------------------
    def handle_line(self, line: str) -> bool:
        """Process one input line. Returns False to request exit."""
        tok = line.strip().split()
        if not tok:
            # Bare Enter: zero-motion hold, keeping the latched gripper.
            self.publish_action([0, 0, 0, 0, 0, 0], self._gripper)
            return True

        head = tok[0].lower()
        if head in {"q", "quit", "exit"}:
            return False
        if head == "help":
            print(_HELP)
            return True
        if head == "reset":
            color = tok[1].lower() if len(tok) > 1 else "blue"
            self.publish_reset(color)
            print(f">> reset -> {color}")
            return True

        # Mnemonic nudge?
        if head in _NUDGES:
            delta, grip = _NUDGES[head]
            gripper = self._gripper if grip is None else grip
            self.publish_action(list(delta), gripper)
            return True

        # Otherwise parse as a raw 7-number action vector.
        try:
            nums = [float(t) for t in tok]
        except ValueError:
            print(f"!! could not parse '{line.strip()}'  (type 'help')")
            return True
        if len(nums) != ri.ACTION_DIM:
            print(f"!! action needs {ri.ACTION_DIM} numbers, got {len(nums)}  (type 'help')")
            return True
        self.publish_action(nums[:6], nums[6])
        return True


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=argv)
    node = TeleopNode()
    print(_HELP)
    print(f">> Publishing to {ri.ACTION_TOPIC}.  Ctrl-D or 'q' to quit.")
    try:
        for line in sys.stdin:
            if not node.handle_line(line):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
