# =============================================================================
# vla_control/control_node.py — Isaac Sim + ROS 2 closed-loop control of the UR10
# =============================================================================
# Boots Isaac Sim, builds the SAME table-top world that vla_collect records, and
# drives the UR10 from 7-DoF actions received on the ROS 2 topic /vla/action —
# the exact action vector OpenVLA emits. Each step it publishes the current
# camera image + proprio state back out, so the action source closes the loop.
#
# Run it through the repo launcher so SimulationApp + the venv + ROS 2 are all
# wired up (see launch.sh / run_control.sh):
#
#     ./run_control.sh --color blue
#   (equivalently: ./launch.sh vla_control/control_node.py --color blue)
#
# Today the action source is vla_control/teleop_node.py (reads actions from
# stdin in the dataset format). Tomorrow, point a trained OpenVLA policy at the
# same /vla/action topic and nothing here changes — the simulator side is
# policy-agnostic by construction.
#
# Boot order matters: SimulationApp MUST be constructed before importing any
# isaacsim.* module, so argument parsing and the app boot happen first and the
# heavy imports come afterwards. rclpy is imported after the app boots too, for
# uniformity (it is safe either way).
# =============================================================================
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_DEBUG = os.environ.get("VLA_DEBUG", "") not in ("", "0")

# Allow `import vla_control.*` / `import vla_collect.*` when run directly as a
# standalone script (only the script's own dir is on sys.path by default).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROS 2 closed-loop control of the UR10 pick-and-place world."
    )
    p.add_argument(
        "--color",
        type=str,
        default=None,
        help="cube colour for the initial instruction (blue/green/yellow/red)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for cube layout")
    p.add_argument(
        "--gui", action="store_true", help="run with the viewport (not headless)"
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace):
    """Construct a ControlConfig from CLI args (config import is Isaac-free)."""
    from vla_control.config import ControlConfig

    cfg = ControlConfig()
    if args.color is not None:
        cfg.initial_color = args.color
    if args.seed is not None:
        cfg.seed = args.seed
    if args.gui:
        cfg.headless = False
    return cfg


# Extensions our scene/controller imports depend on, in dependency order — same
# set the collector enables (the base.python kit app does not auto-enable them).
REQUIRED_EXTENSIONS = (
    "isaacsim.core.api",
    "isaacsim.sensors.camera",
    "isaacsim.robot.manipulators",
    "isaacsim.robot_motion.motion_generation",
    "isaacsim.robot.manipulators.examples",
)


def enable_required_extensions() -> None:
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in REQUIRED_EXTENSIONS:
        mgr.set_extension_enabled_immediate(ext, True)


def configure_render_settings() -> None:
    """Render-quality settings for crisp camera frames (mirrors vla_collect)."""
    import carb

    settings = carb.settings.get_settings()
    settings.set("/rtx/post/aa/op", 2)
    settings.set("/rtx/post/dlss/execMode", 0)
    settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.0)
    settings.set("/app/asyncRendering", False)
    settings.set("/app/asyncRenderingLowLatency", False)
    settings.set("/app/renderer/waitIdle", True)
    settings.set("/rtx/post/histogram/enabled", False)
    settings.set("/rtx/post/tonemap/cameraExposureEnabled", True)
    settings.set("/rtx/post/tonemap/cameraExposure", 12.0)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    # --- boot the simulator FIRST -------------------------------------------
    from isaacsim import SimulationApp

    # RaytracedLighting at construction: gives a crisp frame each step (the
    # default path tracer smears per-step captures). See vla_collect/collect.py.
    sim_app = SimulationApp(
        {"headless": cfg.headless, "renderer": "RaytracedLighting"}
    )

    enable_required_extensions()
    configure_render_settings()

    # Now it is safe to import everything that touches the kit runtime + ROS.
    import rclpy

    from vla_control.controller import VlaControlNode

    rclpy.init(args=None)
    print(f">> Booting ROS 2 control node (headless={cfg.headless})")
    node = VlaControlNode(cfg, sim_app)
    try:
        node.build_and_reset()
        node.run()
    except KeyboardInterrupt:
        print(">> Interrupted; shutting down.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
