# =============================================================================
# vla_collect/config.py — all tunables for the OpenVLA data-collection run
# =============================================================================
# Every magic number the pipeline depends on lives here so the scene, the
# scripted policy, and the dataset schema stay in sync. Nothing in this module
# imports Isaac Sim or ROS, so it can be imported by the offline tools
# (convert_rlds.py, inspect_dataset.py) running under the plain venv python.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- repo-relative paths -----------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_DIR / "data" / "raw"


@dataclass(frozen=True)
class Cube:
    """A coloured cube the arm can pick. `rgb` is linear 0..1 for USD display."""

    name: str
    rgb: tuple[float, float, float]


# The four cubes spawned on the table. Red is included so the policy must
# disambiguate by colour rather than "pick the only coloured thing".
CUBES: tuple[Cube, ...] = (
    Cube("blue", (0.10, 0.20, 0.90)),
    Cube("green", (0.10, 0.75, 0.20)),
    Cube("yellow", (0.95, 0.85, 0.10)),
    Cube("red", (0.90, 0.10, 0.10)),
)


@dataclass(frozen=True)
class WorkspaceConfig:
    """Geometry of the table-top workspace, in metres, in the world frame.

    The UR10 base sits at the world origin on the ground plane (z=0). NVIDIA's
    RMPFlow config for the UR10 is tuned for that base pose and a low working
    plane, so the "table" here is a *thin* slab whose top surface sits just above
    z=0 — the arm reaches out across it rather than standing on a raised bench.
    Everything the arm touches (cubes, pad) rests on that top surface, keeping
    the picking/placing heights inside RMPFlow's reachable range.
    """

    table_thickness: float = 0.02       # thin tabletop slab
    table_center: tuple[float, float] = (0.55, 0.0)
    # Large top so it backs the whole camera frame (otherwise the view above the
    # table is the empty bright background, which washes the image out).
    table_size_xy: tuple[float, float] = (1.60, 2.00)   # x, y extent of the top

    # Cube spawn rectangle (world x,y) — on the tabletop and reachable from origin.
    cube_x: tuple[float, float] = (0.40, 0.68)
    cube_y: tuple[float, float] = (-0.28, 0.28)
    cube_size: float = 0.0515           # edge length (m); matches Isaac demo cube
    min_cube_separation: float = 0.12   # keep spawned cubes from overlapping

    # Red target pad ("rectangle") the cube must be placed on top of. Kept to one
    # side of the spawn band so the arm always traverses the workspace to place.
    pad_center: tuple[float, float] = (0.58, 0.34)
    pad_size: tuple[float, float] = (0.18, 0.14)   # x, y extent (m)
    pad_thickness: float = 0.004

    @property
    def table_top_z(self) -> float:
        """World z of the tabletop surface that cubes and the pad rest on."""
        return self.table_thickness


@dataclass(frozen=True)
class CameraConfig:
    """Third-person camera that produces the OpenVLA observation image.

    OpenVLA consumes a single 224x224 RGB frame, so we render at that resolution
    directly (the SigLIP/DINOv2 backbones expect 224)."""

    prim_path: str = "/World/observation_camera"
    position: tuple[float, float, float] = (1.85, 0.0, 1.35)   # pulled back & up
    look_at: tuple[float, float, float] = (0.45, 0.0, 0.05)    # workspace centre
    resolution: tuple[int, int] = (224, 224)                   # (W, H) saved to disk
    # Render at a larger resolution then downsample to `resolution`. Rendering
    # directly at 224 puts DLSS below its 300px input floor, which upscales from
    # a tiny internal buffer and produces badly blurred frames; 512 avoids that.
    render_resolution: tuple[int, int] = (512, 512)
    focal_length: float = 14.0          # wide-ish FOV so the workspace fits
    frequency: int = 30
    # Soft dome fill added AFTER the scene lights are tamed, so the whole
    # workspace (every cube + the red pad) is lit, not just the spot the robot's
    # end-effector light hits. Modest, so it fills shadows without white-out.
    dome_light_intensity: float = 400.0
    # Global scale applied to every UsdLux light in the stage. The UR10 asset
    # ships a ~9e6-intensity end-effector light and the ground plane a 1e5 sphere
    # light; at full strength they blow the close-up camera out to pure white.
    # 0.03 exposes the matte table and cubes cleanly (calibrated via
    # tools/preview_camera.py). See PickPlaceScene.tame_scene_lights().
    light_intensity_scale: float = 0.03


@dataclass(frozen=True)
class ActionConfig:
    """7-DoF action space, matching OpenVLA's default end-effector control.

    action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
      * the first 6 are the delta of the end-effector pose between consecutive
        recorded steps (world frame, metres / radians),
      * gripper is absolute: 1.0 = closed/grasping, 0.0 = open.
    Deltas are clipped to the bounds below before being written so a few
    outlier frames can't blow up the action normalisation during fine-tuning.
    """

    dim: int = 7
    max_delta_pos: float = 0.05         # m per step, clip
    max_delta_rot: float = 0.20         # rad per step, clip
    # OpenVLA convention used by the Bridge/RT-X data: gripper 1=open, 0=closed
    # is *also* common; we standardise on 1=CLOSED here and document it in meta.
    gripper_closed: float = 1.0
    gripper_open: float = 0.0

    # End-effector offset fed to the RMPFlow pick-place controller so the suction
    # cup contacts the top of the cube. Matches NVIDIA's UR10 cube pick example.
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.02)


@dataclass
class CollectConfig:
    """Top-level configuration for a collection run."""

    # --- dataset shape -------------------------------------------------------
    num_episodes: int = 50
    # The RMPFlow UR10 pick-place state machine takes ~1125 physics steps end to
    # end; the cap is set above that so a full rollout completes, with margin.
    max_steps_per_episode: int = 1400
    record_every_n_physics_steps: int = 3   # decimate 60Hz physics -> ~20Hz data
    seed: int = 0

    # The natural-language instruction template stored with every step. OpenVLA
    # is language-conditioned, so the colour is interpolated per episode.
    instruction_template: str = "place the {color} cube on the red rectangle"

    # --- output --------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    save_video: bool = False            # also dump an mp4 per episode (debug)

    # --- simulation ----------------------------------------------------------
    headless: bool = True
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    # Only keep episodes whose cube ends up on the pad. Failed rollouts are
    # noise for behaviour cloning, so they are discarded by default.
    keep_only_successful: bool = True

    # --- nested configs ------------------------------------------------------
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    def instruction_for(self, color: str) -> str:
        return self.instruction_template.format(color=color)

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


# A module-level default the tools import directly; collect.py builds its own
# from CLI args.
DEFAULT = CollectConfig()
