#!/usr/bin/env python3
# =============================================================================
# tools/convert_rlds.py — recorded episodes -> RLDS/TFDS for OpenVLA fine-tuning
# =============================================================================
# OpenVLA fine-tunes on datasets in the RLDS (Reinforcement Learning Datasets)
# format produced by TFDS — the same layout as Open-X-Embodiment / Bridge. This
# is a standard `tfds.core.GeneratorBasedBuilder` (the pattern from OpenVLA's
# `rlds_dataset_builder`) that turns the .npz episodes written by
# vla_collect/collect.py into that format.
#
# Each RLDS step carries:
#     observation.image            uint8  [H, W, 3]
#     observation.state            float32 [7]   (eef xyz, rpy, gripper)
#     action                       float32 [7]   (delta-eef + abs gripper)
#     language_instruction         string
#     reward / discount / is_first / is_last / is_terminal
#
# TensorFlow + tfds are heavy and intentionally NOT in requirements.txt (the
# collection side only needs numpy/Isaac). Install them just for conversion:
#     .venv/bin/pip install "tensorflow-cpu>=2.15" "tensorflow-datasets>=4.9"
#
# Build the dataset (writes into <out-dir>/<name>/1.0.0/):
#     .venv/bin/python tools/convert_rlds.py --data-dir data/raw --out-dir data/rlds
#
# Then point OpenVLA's fine-tune config at the dataset, or load it directly:
#     import tensorflow_datasets as tfds
#     ds = tfds.builder_from_directory("data/rlds/robo_pickplace/1.0.0").as_dataset()
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vla_collect.recorder import load_episode  # noqa: E402

# Resolved from CLI before the builder is instantiated (tfds builders take no
# constructor args for custom data). Module-level so the builder class sees it.
_SOURCE_DIR: Path = Path("data/raw")
_IMAGE_HW: tuple[int, int] = (224, 224)
_DATASET_NAME: str = "robo_pickplace"


def _episode_to_rlds(npz_path: Path) -> dict:
    """Build one RLDS episode dict (a list of steps + metadata) from an .npz."""
    ep = load_episode(npz_path)
    images = ep["images"]
    states = ep["states"].astype(np.float32)
    actions = ep["actions"].astype(np.float32)
    meta = ep.get("meta", {})
    instruction = meta.get("instruction", "place the cube on the red rectangle")
    success = bool(meta.get("success", False))
    n = images.shape[0]

    steps = []
    for i in range(n):
        last = i == (n - 1)
        steps.append(
            {
                "observation": {
                    "image": images[i],
                    "state": states[i],
                },
                "action": actions[i],
                "discount": np.float32(1.0),
                # Sparse terminal reward; OpenVLA's BC ignores it but RLDS keeps it.
                "reward": np.float32(1.0 if (last and success) else 0.0),
                "is_first": i == 0,
                "is_last": last,
                "is_terminal": last,
                "language_instruction": instruction,
            }
        )
    return {
        "steps": steps,
        "episode_metadata": {"file_path": str(npz_path), "success": success},
    }


def _make_builder_class():
    """Define the GeneratorBasedBuilder lazily (only when TF is importable)."""
    import tensorflow_datasets as tfds

    h, w = _IMAGE_HW

    class RoboPickPlace(tfds.core.GeneratorBasedBuilder):
        """UR10 colour-cube pick-and-place episodes for OpenVLA fine-tuning."""

        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Initial Isaac Sim 6.0.0 collection."}
        # Let tfds write under our chosen name regardless of the class name.
        name = _DATASET_NAME

        def _info(self) -> "tfds.core.DatasetInfo":
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(h, w, 3),
                                            dtype=np.uint8,
                                            encoding_format="png",
                                            doc="Third-person 224x224 RGB observation.",
                                        ),
                                        "state": tfds.features.Tensor(
                                            shape=(7,),
                                            dtype=np.float32,
                                            doc="EEF xyz, roll/pitch/yaw, gripper.",
                                        ),
                                    }
                                ),
                                "action": tfds.features.Tensor(
                                    shape=(7,),
                                    dtype=np.float32,
                                    doc="Delta EEF (dx..dyaw) + absolute gripper.",
                                ),
                                "discount": tfds.features.Scalar(dtype=np.float32),
                                "reward": tfds.features.Scalar(dtype=np.float32),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "file_path": tfds.features.Text(),
                                "success": tfds.features.Scalar(dtype=np.bool_),
                            }
                        ),
                    }
                ),
            )

        def _split_generators(self, dl_manager):
            # All episodes go to the train split; OpenVLA fine-tunes on train.
            return {"train": self._generate_examples(sorted(_SOURCE_DIR.glob("episode_*.npz")))}

        def _generate_examples(self, paths):
            for p in paths:
                yield p.stem, _episode_to_rlds(Path(p))

    return RoboPickPlace


def build(data_dir: Path, out_dir: Path, name: str, image_hw: tuple[int, int]) -> None:
    global _SOURCE_DIR, _IMAGE_HW, _DATASET_NAME
    _SOURCE_DIR = data_dir
    _IMAGE_HW = image_hw
    _DATASET_NAME = name

    try:
        import tensorflow_datasets as tfds  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "TensorFlow + tensorflow-datasets are required for RLDS conversion.\n"
            '  .venv/bin/pip install "tensorflow-cpu>=2.15" "tensorflow-datasets>=4.9"\n'
            f"(import failed: {e})"
        )

    if not sorted(data_dir.glob("episode_*.npz")):
        raise SystemExit(f"No episodes found in {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    builder_cls = _make_builder_class()
    builder = builder_cls(data_dir=str(out_dir))
    builder.download_and_prepare()
    final = out_dir / name / str(builder.version)
    print(f">> Wrote RLDS dataset '{name}' to {final}")
    print(">> Load in OpenVLA with:")
    print(f"     tfds.builder_from_directory('{final}').as_dataset()")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Convert recorded episodes to an RLDS/TFDS dataset for OpenVLA."
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/rlds"))
    ap.add_argument("--name", type=str, default="robo_pickplace")
    ap.add_argument("--image-height", type=int, default=224)
    ap.add_argument("--image-width", type=int, default=224)
    args = ap.parse_args(argv)
    build(args.data_dir, args.out_dir, args.name, (args.image_height, args.image_width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
