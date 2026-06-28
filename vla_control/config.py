# =============================================================================
# vla_control/config.py — tunables for the ROS 2 closed-loop control node
# =============================================================================
# Reuses the *same* world geometry, camera, and action conventions as the data
# collector (vla_collect.config) so the simulator the policy controls is bit-for-
# bit the one it was trained on. This module only adds the few knobs that are
# specific to live, action-driven control. Isaac-free, like vla_collect.config.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field

# Reuse the collector's world/camera/action configuration verbatim — the whole
# point is that control happens in the identical environment.
from vla_collect.config import (
    CUBES,
    ActionConfig,
    CameraConfig,
    WorkspaceConfig,
)

__all__ = ["CUBES", "ActionConfig", "CameraConfig", "WorkspaceConfig", "ControlConfig"]


@dataclass
class ControlConfig:
    """Top-level configuration for the ROS 2 control node."""

    # --- task ----------------------------------------------------------------
    # The instruction published on /vla/instruction at startup. Matches the
    # template vla_collect trains against; `--color` on the CLI fills the blank.
    instruction_template: str = "place the {color} cube on the red rectangle"
    initial_color: str = "blue"

    # --- simulation ----------------------------------------------------------
    headless: bool = True
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    seed: int = 0

    # --- control loop --------------------------------------------------------
    # Actions arrive as *delta* end-effector poses. We integrate each delta onto
    # a running target pose and let RMPFlow servo the arm toward it. To match the
    # ~20 Hz rate the dataset was decimated to, we hold each received action for
    # `physics_steps_per_action` physics ticks (60 Hz / 3 = 20 Hz) before the
    # target is allowed to advance again — i.e. one action == one dataset step.
    physics_steps_per_action: int = 3

    # If no new action arrives, keep servoing toward the last target (the arm
    # finishes the in-flight motion) rather than freezing mid-trajectory.
    hold_last_target: bool = True

    # Safety clamp: the integrated target EE position is kept inside this axis-
    # aligned box (world frame, metres) so a runaway stream of deltas can't drive
    # the solver to an unreachable pose. Sized to the reachable workspace + head-
    # room above the table.
    target_x: tuple[float, float] = (0.25, 0.85)
    target_y: tuple[float, float] = (-0.45, 0.45)
    target_z: tuple[float, float] = (0.05, 0.65)

    # --- observation publishing ----------------------------------------------
    # Publish an (image, state) observation every N physics steps. Defaults to
    # the dataset's record cadence so the live stream matches training data rate.
    publish_every_n_physics_steps: int = 3

    # --- nested configs (shared with vla_collect) ----------------------------
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    def instruction_for(self, color: str) -> str:
        return self.instruction_template.format(color=color)
