# =============================================================================
# vla_collect/scene.py — build the table-top pick-and-place world
# =============================================================================
# Must be imported only AFTER SimulationApp() has been constructed (see
# collect.py) — the isaacsim.* modules require a running kit app.
#
# Layout (world frame, metres):
#   * a UR10 6-axis arm with its base at the world origin on the ground (z=0),
#   * a thin fixed tabletop slab the arm reaches across (top at table_top_z),
#   * a thin red rectangle ("the red rectangle") = the placement target,
#   * N coloured dynamic cubes randomised within the spawn rectangle,
#   * a fixed third-person camera that renders the 224x224 OpenVLA observation.
#
# The UR10 base is at the origin (not raised on the table) because NVIDIA's
# bundled RMPFlow config that drives the scripted pick-place is tuned for that
# base pose; raising the robot pushes targets outside its solved workspace.
# =============================================================================
from __future__ import annotations

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.universal_robots import UR10
from isaacsim.robot.manipulators.examples.universal_robots.controllers import (
    PickPlaceController,
)
from isaacsim.sensors.camera import Camera

from .config import CUBES, CollectConfig


class PickPlaceScene:
    """Owns every prim in the world and exposes a tidy handle for the policy."""

    def __init__(self, cfg: CollectConfig) -> None:
        self.cfg = cfg
        self.ws = cfg.workspace
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=cfg.physics_dt,
            rendering_dt=cfg.rendering_dt,
        )
        self.robot: UR10 | None = None
        self.controller: PickPlaceController | None = None
        self.camera: Camera | None = None
        self.cubes: dict[str, DynamicCuboid] = {}

    # --- construction --------------------------------------------------------
    def build(self) -> None:
        """Populate the stage. Call once before world.reset()."""
        self.world.scene.add_default_ground_plane()
        self._add_table()
        self._add_red_pad()
        self._add_robot()
        self._add_cubes()
        self._add_camera()
        # Dim the over-bright default/robot lights, THEN add soft fill so the
        # whole workspace (all cubes + red pad) is evenly lit without white-out.
        self.tame_scene_lights()
        self._add_fill_light()

    def _add_fill_light(self) -> None:
        # Soft, even fill so the whole workspace (all cubes + the red pad) is
        # visible, not just whatever the robot's harsh end-effector light hits.
        # Added AFTER tame_scene_lights() so it isn't scaled down with the rest.
        intensity = self.cfg.camera.dome_light_intensity
        if intensity > 0:
            from isaacsim.core.experimental.objects import DomeLight

            DomeLight("/World/FillDomeLight").set_intensities(intensity)

    def tame_scene_lights(self) -> None:
        """Dim the over-bright default lights so the camera isn't blown out.

        The UR10 asset ships with a ~9e6-intensity light parented to its
        end-effector (a viewport convenience) and the default ground plane adds a
        very strong sphere light. Both saturate our close-up camera to pure
        white, so we scale every UsdLux light by a single factor (preserving
        their relative balance). Called after build(), once the robot/ground
        prims exist on the stage.
        """
        import omni.usd
        from pxr import UsdLux

        scale = self.cfg.camera.light_intensity_scale
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            attr = UsdLux.LightAPI(prim).GetIntensityAttr()
            attr.Set((attr.Get() or 0.0) * scale)

    def _add_table(self) -> None:
        sx, sy = self.ws.table_size_xy
        cx, cy = self.ws.table_center
        sz = self.ws.table_thickness
        # Thin fixed slab; its centre sits at sz/2 so the top surface is at
        # table_top_z. A FixedCuboid is a static collider so cubes rest on it.
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/table",
                name="table",
                position=np.array([cx, cy, sz / 2.0]),
                scale=np.array([sx, sy, sz]),
                color=np.array([0.45, 0.32, 0.22]),
            )
        )

    def _add_red_pad(self) -> None:
        px, py = self.ws.pad_center
        ex, ey = self.ws.pad_size
        # A thin, *visual-only* red slab resting on the table. It is the goal
        # marker, not an obstacle, so a VisualCuboid (no collider) is correct —
        # the cube must rest on the table at the pad's xy, not balance on a lip.
        self.world.scene.add(
            VisualCuboid(
                prim_path="/World/red_rectangle",
                name="red_rectangle",
                position=np.array([px, py, self.ws.table_top_z + self.ws.pad_thickness / 2.0]),
                scale=np.array([ex, ey, self.ws.pad_thickness]),
                color=np.array([0.9, 0.08, 0.08]),
            )
        )

    def _add_robot(self) -> None:
        # UR10 with the default short-suction surface gripper, base at the world
        # origin on the ground plane (z=0) — see module docstring.
        robot = UR10(
            prim_path="/World/UR10",
            name="ur10",
            position=np.array([0.0, 0.0, 0.0]),
            attach_gripper=True,
        )
        # A reasonable "ready" pose hovering over the workspace (matches the
        # joint defaults NVIDIA's UR10 pick-place examples start from).
        robot.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0.0])
        )
        self.robot = self.world.scene.add(robot)
        # Suction gripper starts open so the first pick can attach.
        self.robot.gripper.set_default_state(opened=True)

    def _add_cubes(self) -> None:
        z = self.ws.table_top_z + self.ws.cube_size / 2.0
        for cube in CUBES:
            prim_path = f"/World/cube_{cube.name}"
            obj = DynamicCuboid(
                prim_path=prim_path,
                name=f"cube_{cube.name}",
                position=np.array([0.6, 0.0, z]),   # real pose set in randomise()
                scale=np.array([self.ws.cube_size] * 3),
                color=np.array(cube.rgb),
                mass=0.05,
            )
            self.cubes[cube.name] = self.world.scene.add(obj)

    def _add_camera(self) -> None:
        cc = self.cfg.camera
        pos = np.array(cc.position, dtype=np.float64)
        self.camera = Camera(
            prim_path=cc.prim_path,
            position=pos,
            orientation=self._look_at_quat(pos, np.array(cc.look_at, dtype=np.float64)),
            frequency=cc.frequency,
            resolution=cc.render_resolution,   # render large; downsample on capture
        )

    def aim_camera(self, eye, look_at) -> None:
        """Re-pose the observation camera at runtime (used by preview tooling)."""
        eye = np.asarray(eye, dtype=np.float64)
        quat = self._look_at_quat(eye, np.asarray(look_at, dtype=np.float64))
        self.camera.set_world_pose(position=eye, orientation=quat)

    @staticmethod
    def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        """USD-convention quaternion (wxyz) aiming the camera from eye at target.

        Derived from the yaw/pitch of the eye->target vector. This matches the
        orientation form the camera constructor expects (the same construction
        that produced correctly-framed frames in testing).
        """
        direction = target - eye
        yaw = np.arctan2(direction[1], direction[0])
        pitch = np.arctan2(direction[2], np.linalg.norm(direction[:2]))
        return euler_angles_to_quat(np.array([0.0, -pitch, yaw]))

    # --- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        """Reset physics; initialises robot, gripper, camera annotators."""
        self.world.reset()
        if self.camera is not None:
            self.camera.initialize()
            self.camera.set_focal_length(self.cfg.camera.focal_length)
            # The UR10 camera prim ships with a non-standard ~2.1mm horizontal
            # aperture; without this the focal length acts as an extreme
            # telephoto and the arm/cubes fall outside the frame. Force the
            # standard 35mm-film aperture so focal_length sets a sane FOV.
            self.camera.set_horizontal_aperture(self.cfg.camera.horizontal_aperture)
            # fStop = 0 disables depth-of-field, giving a sharp pinhole image.
            # A non-zero default fStop with the wrong focus distance is what made
            # the observation frames look smeared/out of focus.
            self.camera.set_lens_aperture(0.0)
            # Attach the depth / pointcloud annotators (after initialize(), which
            # creates the render product they hang off). distance_to_image_plane
            # gives metric depth aligned with the RGB frame; the pointcloud
            # annotator gives an (N,3) cloud. A few render steps are needed before
            # they return valid data — the per-episode warm-up loop covers that.
            cc = self.cfg.camera
            if cc.enable_depth:
                self.camera.add_distance_to_image_plane_to_frame()
            if cc.enable_pointcloud:
                self.camera.add_pointcloud_to_frame()
        # Build the scripted RMPFlow pick-place controller now that the
        # articulation exists and is initialised.
        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot,
        )

    # --- per-episode randomisation ------------------------------------------
    def randomise_cubes(self, rng: np.random.Generator) -> None:
        """Place every cube at a fresh, non-overlapping spot in the spawn rect."""
        z = self.ws.table_top_z + self.ws.cube_size / 2.0
        placed: list[np.ndarray] = []
        for name, obj in self.cubes.items():
            xy = self._sample_free_xy(rng, placed)
            placed.append(xy)
            pos = np.array([xy[0], xy[1], z])
            obj.set_world_pose(position=pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            # Zero any residual velocity from the previous episode.
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

    def _sample_free_xy(self, rng: np.random.Generator, placed: list[np.ndarray]) -> np.ndarray:
        for _ in range(200):
            x = rng.uniform(*self.ws.cube_x)
            y = rng.uniform(*self.ws.cube_y)
            xy = np.array([x, y])
            if all(np.linalg.norm(xy - p) >= self.ws.min_cube_separation for p in placed):
                return xy
        # Fallback: accept the last sample rather than loop forever.
        return xy

    # --- observation helpers -------------------------------------------------
    def capture_image(self) -> np.ndarray:
        """Return the camera RGB downsampled to the saved resolution as uint8.

        The sensor renders at `render_resolution` (to keep DLSS happy); we area-
        resample that down to the `resolution` OpenVLA consumes, or return zeros
        if the renderer hasn't produced a frame yet.
        """
        w, h = self.cfg.camera.resolution
        rgba = self.camera.get_rgba() if self.camera is not None else None
        if rgba is None or rgba.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        rgb = np.ascontiguousarray(rgba[..., :3]).astype(np.uint8)
        if rgb.shape[:2] != (h, w):
            from PIL import Image

            rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        return rgb

    def capture_depth(self) -> np.ndarray | None:
        """Return per-pixel metric depth (distance to image plane) in metres.

        Shape is the camera's *render* resolution (H, W) float32 — NOT downsampled
        to the 224x224 RGB observation, since depth nearest-neighbour resampling
        would corrupt edges; downstream consumers resize as they see fit. Returns
        None if depth is disabled or the annotator has no frame yet (e.g. before
        the renderer has warmed up).
        """
        if self.camera is None or not self.cfg.camera.enable_depth:
            return None
        depth = self.camera.get_depth()
        if depth is None or depth.size == 0:
            return None
        return np.ascontiguousarray(depth).astype(np.float32)

    def capture_pointcloud(self) -> np.ndarray | None:
        """Return an (N, 3) float32 pointcloud from the camera.

        Points are in the world frame when `camera.pointcloud_world_frame` is
        True, else the camera frame. Returns None if pointcloud is disabled or no
        valid frame is available yet.
        """
        if self.camera is None or not self.cfg.camera.enable_pointcloud:
            return None
        pts = self.camera.get_pointcloud(world_frame=self.cfg.camera.pointcloud_world_frame)
        if pts is None or len(pts) == 0:
            return None
        return np.ascontiguousarray(pts).astype(np.float32).reshape(-1, 3)

    def end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World (position xyz, orientation wxyz) of the gripper end-effector."""
        return self.robot.end_effector.get_world_pose()

    def pad_target_xy(self) -> np.ndarray:
        return np.array(self.ws.pad_center, dtype=np.float64)

    def cube_position(self, color: str) -> np.ndarray:
        pos, _ = self.cubes[color].get_world_pose()
        return np.asarray(pos, dtype=np.float64)

    def is_on_pad(self, color: str) -> bool:
        """Success test: cube centre over the pad rectangle and resting on it."""
        pos = self.cube_position(color)
        px, py = self.ws.pad_center
        ex, ey = self.ws.pad_size
        within_xy = abs(pos[0] - px) <= ex / 2.0 and abs(pos[1] - py) <= ey / 2.0
        rest_z = self.ws.table_top_z + self.ws.cube_size / 2.0
        on_surface = abs(pos[2] - rest_z) <= 0.03
        return bool(within_xy and on_surface)
