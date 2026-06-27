#!/usr/bin/env python3
# =============================================================================
# tools/inspect_dataset.py — sanity-check a recorded collection run
# =============================================================================
# Reads the .npz/.json episodes written by vla_collect/collect.py and prints a
# summary: episode count, success rate, per-colour balance, step-length stats,
# action ranges, and (optionally) dumps a contact sheet of the first frame of
# each episode so you can eyeball the camera view.
#
# Pure numpy + Pillow — runs under the plain venv python, no Isaac/TF needed:
#     .venv/bin/python tools/inspect_dataset.py data/raw
#     .venv/bin/python tools/inspect_dataset.py data/raw --contact-sheet sheet.png
# =============================================================================
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Make `vla_collect` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vla_collect.recorder import load_episode  # noqa: E402


def find_episodes(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("episode_*.npz"))


def summarise(data_dir: Path) -> dict:
    paths = find_episodes(data_dir)
    if not paths:
        raise SystemExit(f"No episodes found in {data_dir}")

    lengths: list[int] = []
    colors: Counter = Counter()
    successes = 0
    act_min = np.full(7, np.inf)
    act_max = np.full(7, -np.inf)
    img_shape = None

    for p in paths:
        ep = load_episode(p)
        meta = ep.get("meta", {})
        n = int(ep["images"].shape[0])
        lengths.append(n)
        colors[meta.get("color", "?")] += 1
        successes += int(bool(meta.get("success", False)))
        acts = ep["actions"]
        act_min = np.minimum(act_min, acts.min(axis=0))
        act_max = np.maximum(act_max, acts.max(axis=0))
        if img_shape is None:
            img_shape = tuple(ep["images"].shape[1:])

    lengths_arr = np.array(lengths)
    return {
        "data_dir": str(data_dir),
        "num_episodes": len(paths),
        "success_rate": successes / len(paths),
        "color_balance": dict(colors),
        "steps_min": int(lengths_arr.min()),
        "steps_max": int(lengths_arr.max()),
        "steps_mean": float(lengths_arr.mean()),
        "total_steps": int(lengths_arr.sum()),
        "image_shape": img_shape,
        "action_min": act_min.round(4).tolist(),
        "action_max": act_max.round(4).tolist(),
    }


def print_summary(s: dict) -> None:
    print(f"Dataset: {s['data_dir']}")
    print(f"  episodes      : {s['num_episodes']}")
    print(f"  success rate  : {s['success_rate']:.1%}")
    print(f"  colour balance: {s['color_balance']}")
    print(f"  steps         : min={s['steps_min']} mean={s['steps_mean']:.1f} "
          f"max={s['steps_max']} total={s['total_steps']}")
    print(f"  image shape   : {s['image_shape']}")
    layout = ["dx", "dy", "dz", "dR", "dP", "dY", "grip"]
    print("  action ranges :")
    for name, lo, hi in zip(layout, s["action_min"], s["action_max"]):
        print(f"      {name:>4}: [{lo:+.4f}, {hi:+.4f}]")


def contact_sheet(data_dir: Path, out: Path, per_row: int = 8) -> None:
    """Tile the first frame of every episode into one PNG for a quick eyeball."""
    from PIL import Image

    paths = find_episodes(data_dir)
    frames = []
    for p in paths:
        with np.load(p) as d:
            frames.append(d["images"][0])
    h, w = frames[0].shape[:2]
    rows = (len(frames) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * w, rows * h), (20, 20, 20))
    for i, fr in enumerate(frames):
        r, c = divmod(i, per_row)
        sheet.paste(Image.fromarray(fr), (c * w, r * h))
    sheet.save(out)
    print(f">> wrote contact sheet: {out}  ({len(frames)} frames)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Inspect a recorded OpenVLA collection run.")
    ap.add_argument("data_dir", type=Path, help="directory of episode_*.npz files")
    ap.add_argument("--contact-sheet", type=Path, default=None,
                    help="write a tiled PNG of each episode's first frame")
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = ap.parse_args(argv)

    s = summarise(args.data_dir)
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print_summary(s)
    if args.contact_sheet:
        contact_sheet(args.data_dir, args.contact_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
