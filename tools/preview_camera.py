#!/usr/bin/env python3
# =============================================================================
# tools/preview_camera.py — dump a single observation frame (no full episode)
# =============================================================================
# Builds the scene, randomises the cubes, warms up the renderer, and saves one
# observation frame to a PNG. Handy for checking camera framing / lighting after
# editing config.py without paying for a full ~20s pick-place rollout.
#
#     ./launch.sh tools/preview_camera.py --out /tmp/preview.png
#
# The camera pose is taken from CameraConfig; edit config.py and re-run to
# iterate on framing. (We deliberately do not re-pose the camera at runtime —
# Camera.set_world_pose after construction can detach the render product.)
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/tmp/preview.png")
    ap.add_argument("--warmup", type=int, default=20, help="render steps before capture")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

    from vla_collect.collect import configure_render_settings, enable_required_extensions

    enable_required_extensions()
    configure_render_settings()

    import numpy as np
    from PIL import Image

    from vla_collect.config import CollectConfig
    from vla_collect.scene import PickPlaceScene

    cfg = CollectConfig()
    scene = PickPlaceScene(cfg)
    scene.build()
    scene.reset()
    scene.randomise_cubes(np.random.default_rng(args.seed))

    for _ in range(args.warmup):
        scene.world.step(render=True)

    img = scene.capture_image()
    Image.fromarray(img).save(args.out)
    print(f">> wrote {args.out}  mean={img.mean():.1f} std={img.std():.1f} shape={img.shape}")
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
