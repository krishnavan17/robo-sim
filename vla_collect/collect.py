# =============================================================================
# vla_collect/collect.py — standalone Isaac Sim entry point for data collection
# =============================================================================
# Runs the UR10 pick-and-place task repeatedly under a scripted RMPFlow policy,
# records each rollout as an OpenVLA-shaped episode, and writes them to disk.
#
# Run it through the repo's launcher so SimulationApp + the venv are wired up:
#
#     ./run_collect.sh --num-episodes 100 --data-dir data/raw
#   (equivalently: ./launch.sh vla_collect/collect.py --num-episodes 100)
#
# SimulationApp MUST be constructed before importing any isaacsim.* module, so
# argument parsing and the app boot happen at the very top, and the heavy
# imports (scene.py) come afterwards.
# =============================================================================
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_DEBUG = os.environ.get("VLA_DEBUG", "") not in ("", "0")

# Allow `import vla_collect.*` when this file is run directly as a script
# (./launch.sh vla_collect/collect.py) — only the script's own dir is on the
# path by default, so add the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect UR10 pick-and-place episodes for OpenVLA fine-tuning."
    )
    p.add_argument("--num-episodes", type=int, default=None, help="episodes to record")
    p.add_argument("--data-dir", type=str, default=None, help="output directory")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--max-steps", type=int, default=None, help="step cap per episode")
    p.add_argument("--gui", action="store_true", help="run with the viewport (not headless)")
    p.add_argument(
        "--keep-failures",
        action="store_true",
        help="also save rollouts that don't end on the pad",
    )
    p.add_argument(
        "--publish",
        action="store_true",
        help="also mirror each recorded frame onto the vla_control ROS 2 topics",
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace):
    """Construct a CollectConfig from CLI args (config import is Isaac-free)."""
    from vla_collect.config import CollectConfig

    cfg = CollectConfig()
    if args.num_episodes is not None:
        cfg.num_episodes = args.num_episodes
    if args.data_dir is not None:
        cfg.data_dir = Path(args.data_dir)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_steps is not None:
        cfg.max_steps_per_episode = args.max_steps
    if args.gui:
        cfg.headless = False
    if args.keep_failures:
        cfg.keep_only_successful = False
    return cfg


# Extensions our scene/controller imports depend on, in dependency order. The
# base.python kit app boots without these, so we enable them explicitly once the
# SimulationApp exists. set_extension_enabled_immediate pulls in dependencies.
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
    """Render-quality settings for crisp synthetic camera frames.

    DLSS in its default (performance) mode renders from a small internal buffer
    and upscales, which blurs our observation. execMode=2 selects the quality
    preset; we also disable async rendering so each world.step() returns a fully
    converged frame (important for deterministic, sharp data capture).
    """
    import carb

    settings = carb.settings.get_settings()
    # NB: the render MODE (RaytracedLighting) is set at SimulationApp construction
    # — it cannot be changed here via carb. See the SimulationApp call in main().
    # Antialiasing: 0=off 1=TAA 2=FXAA 3=DLSS 4=DLAA. FXAA is a sharp single-pass
    # filter (DLSS upscales from a small buffer and blurs small frames).
    settings.set("/rtx/post/aa/op", 2)
    settings.set("/rtx/post/dlss/execMode", 0)
    # No motion blur — it smears moving cubes/arm across the captured frame.
    settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.0)
    # Synchronous rendering (asyncRendering=False) already guarantees each
    # world.step(render=True) returns a complete frame before capture. waitIdle
    # additionally forces a FULL GPU pipeline drain every step, which serialises
    # CPU and GPU and leaves the GPU idle most of the frame (slow run, low GPU
    # utilisation). Synchronous rendering alone is enough for valid captures, so
    # leave waitIdle off and let the driver pipeline normally.
    settings.set("/app/asyncRendering", False)
    settings.set("/app/asyncRenderingLowLatency", False)
    settings.set("/app/renderer/waitIdle", False)
    # Disable histogram auto-exposure: it adapts to the bright sky and blows the
    # table-top out to white. Fixed exposure keeps frames stable across the run.
    settings.set("/rtx/post/histogram/enabled", False)
    settings.set("/rtx/post/tonemap/cameraExposureEnabled", True)
    settings.set("/rtx/post/tonemap/cameraExposure", 12.0)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    # SimulationApp reads sys.argv directly and forwards anything it doesn't
    # recognise to the underlying kit application. Our flags (e.g. --publish,
    # which collides with a built-in kit flag that expects a value) would crash
    # kit's arg parser, so scrub argv down to just the program name now that we
    # have parsed what we need.
    sys.argv = sys.argv[:1]

    # --- boot the simulator FIRST -------------------------------------------
    from isaacsim import SimulationApp

    # renderer="RaytracedLighting" must be set at construction: it's baked into
    # the kit experience and cannot be changed via carb settings afterwards. The
    # default RealTimePathTracing accumulates samples across frames, which smears
    # our per-step captures; RaytracedLighting gives a crisp frame every step.
    sim_app = SimulationApp({"headless": cfg.headless, "renderer": "RaytracedLighting"})

    # The UR10 manipulator + RMPFlow stack and the camera sensor live in
    # extensions the base kit app doesn't auto-enable (several ship under
    # extsDeprecated/). Enable them immediately, before importing scene.py.
    enable_required_extensions()
    configure_render_settings()

    # Now it is safe to import everything that touches the kit runtime.
    import numpy as np

    from vla_collect.config import CUBES
    from vla_collect.recorder import EpisodeRecorder
    from vla_collect.scene import PickPlaceScene

    # Depth + pointcloud annotators add GPU work to every rendered frame, but
    # they are only consumed by the --publish mirror. For plain RGB collection
    # they are pure render overhead, so disable them unless we're publishing.
    if not args.publish and (cfg.camera.enable_depth or cfg.camera.enable_pointcloud):
        import dataclasses

        cfg.camera = dataclasses.replace(
            cfg.camera, enable_depth=False, enable_pointcloud=False
        )

    rng = cfg.rng()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    print(f">> Building scene (headless={cfg.headless})")
    scene = PickPlaceScene(cfg)
    scene.build()
    scene.reset()

    # Optional ROS 2 mirror: re-publish every recorded frame onto the SAME topics
    # vla_control serves, so a live consumer sees an identical stream from the
    # scripted demos. rclpy/the publisher are imported only when --publish is set.
    publisher = None
    if args.publish:
        import rclpy

        from vla_collect.publisher import ObservationPublisher

        rclpy.init()
        publisher = ObservationPublisher(cfg)
        print(">> Publishing observations on the /vla/observation/* topics")

    # Cube colour cycles deterministically across episodes so the dataset is
    # balanced over the four instructions.
    colors = [c.name for c in CUBES]

    saved = 0
    attempted = 0
    episode_index = 0
    while saved < cfg.num_episodes:
        color = colors[attempted % len(colors)]
        attempted += 1
        instruction = cfg.instruction_for(color)
        rec = EpisodeRecorder(instruction=instruction, color=color, action_cfg=cfg.action)

        if publisher is not None:
            publisher.publish_instruction(instruction)

        success = run_episode(cfg, scene, rec, color, rng, publisher)
        rec.set_success(success)

        if success or not cfg.keep_only_successful:
            npz = rec.save(cfg.data_dir, episode_index)
            episode_index += 1
            saved += 1
            tag = "ok " if success else "FAIL"
            print(f">> [{tag}] saved {npz.name}  ({len(rec)} steps, '{instruction}')  "
                  f"[{saved}/{cfg.num_episodes}]")
        else:
            print(f">> [drop] '{instruction}' did not reach the pad; discarded "
                  f"(attempt {attempted})")

        # Safety valve: don't loop forever if the policy keeps failing. Allow a
        # generous number of attempts per requested episode before giving up.
        if attempted >= max(20, cfg.num_episodes * 5) and saved == 0:
            print("!! No successful episodes after many attempts — aborting.", file=sys.stderr)
            break

    print(f">> Done. Wrote {saved} episodes to {cfg.data_dir}")
    if publisher is not None:
        import rclpy

        publisher.destroy()
        rclpy.shutdown()
    sim_app.close()
    return 0


def run_episode(cfg, scene, rec, color: str, rng, publisher=None) -> bool:
    """Run one scripted pick-and-place rollout, recording observations.

    Returns True if the target cube ends up on the red rectangle.
    """
    import numpy as np

    world = scene.world
    # Fresh layout + controller state for this rollout.
    world.reset()
    scene.randomise_cubes(rng)
    scene.controller.reset()

    # NOTE: render=True ALWAYS — even headless. "headless" only suppresses the
    # GUI window; the RTX renderer must still run each step or the observation
    # camera produces no frames (the blank-image / "annotator returned None" bug).
    ee_offset = np.array(cfg.action.ee_offset)

    # Let physics settle and the camera warm up before recording. Only the
    # final few steps need rendering (to prime the camera annotators for the
    # first capture); settling physics doesn't, so skip the RTX path on the
    # rest — rendering is the expensive GPU stage.
    warmup_steps = int(0.3 / cfg.physics_dt)
    for i in range(warmup_steps):
        world.step(render=(i >= warmup_steps - 3))

    cube_z = scene.ws.table_top_z + scene.ws.cube_size / 2.0
    px, py = scene.pad_target_xy()
    placing_position = np.array([px, py, cube_z])

    step = 0
    while step < cfg.max_steps_per_episode:
        # Re-read the live cube position so the grasp tracks any settling drift.
        live = scene.cube_position(color)
        picking_position = np.array([live[0], live[1], cube_z])

        joints = scene.robot.get_joint_positions()
        action = scene.controller.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=joints,
            end_effector_offset=ee_offset,
        )
        scene.robot.apply_action(action)

        # Render ONLY on steps we actually capture. We decimate physics down to
        # ~20Hz data, so 2 of every 3 frames were rendered and thrown away —
        # pure GPU waste. Advance physics with render=False on the dropped steps
        # and run the RTX path only when this step's frame will be recorded.
        capture = step % cfg.record_every_n_physics_steps == 0
        world.step(render=capture)

        # Decimate: record every Nth physics step into the episode buffer.
        if capture:
            scene.robot.gripper.update()
            image = scene.capture_image()
            eef_pos, eef_quat = scene.end_effector_pose()
            gripper_closed = scene.robot.gripper.is_closed()
            rec.add(
                image=image,
                eef_pos=eef_pos,
                eef_quat_wxyz=eef_quat,
                gripper_closed=gripper_closed,
            )
            if publisher is not None:
                publisher.publish_observation(
                    image=image,
                    eef_pos=eef_pos,
                    eef_quat_wxyz=eef_quat,
                    gripper_closed=gripper_closed,
                    depth=scene.capture_depth(),
                    pointcloud=scene.capture_pointcloud(),
                )

        if scene.controller.is_done():
            break
        step += 1

    if _DEBUG:
        cube = scene.cube_position(color)
        eef_pos, _ = scene.end_effector_pose()
        print(f"   [dbg] {color}: steps={step} event={scene.controller.get_current_event()} "
              f"done={scene.controller.is_done()} grip_closed={scene.robot.gripper.is_closed()} "
              f"cube={np.round(cube,3)} eef={np.round(eef_pos,3)} "
              f"pad={np.round(scene.pad_target_xy(),3)}", file=sys.stderr)

    # Let the cube settle, then evaluate success. No frames are captured here,
    # so physics-only steps (render=False) are enough — and much faster.
    for _ in range(int(0.5 / cfg.physics_dt)):
        world.step(render=False)
    return scene.is_on_pad(color)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
