# =============================================================================
# vla_collect/recorder.py — episode buffer and on-disk format (pure numpy)
# =============================================================================
# Accumulates one pick-and-place rollout as a list of timesteps, derives the
# 7-DoF end-effector *delta* actions OpenVLA trains on, and writes each episode
# to a self-describing .npz plus a JSON sidecar. No Isaac Sim / ROS imports, so
# this is unit-testable and importable by the offline conversion tools.
#
# OpenVLA training expects, per timestep:
#   observation.image   : uint8 [H, W, 3]   third-person RGB
#   observation.state    : float32 [7]       proprio (EEF xyz + rpy + gripper)
#   action               : float32 [7]       delta-EEF (dx..dyaw) + abs gripper
#   language_instruction : str
# We store states for every recorded frame and compute action[t] as the
# transition state[t] -> state[t+1]; the final frame repeats a zero-motion
# action with the terminal gripper command so episode length stays consistent.
# =============================================================================
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ActionConfig


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to roll/pitch/yaw (XYZ extrinsic).

    Isaac Sim returns orientations as wxyz quaternions; OpenVLA's state/action
    use Euler angles, so we flatten here. Pitch is clamped to avoid a NaN at the
    gimbal singularity.
    """
    w, x, y, z = quat_wxyz
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi] so rotation deltas take the short way round."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class _Step:
    image: np.ndarray          # uint8 [H, W, 3]
    eef_pos: np.ndarray        # float64 [3] world xyz
    eef_rpy: np.ndarray        # float64 [3] roll/pitch/yaw
    gripper: float             # absolute command, 1=closed 0=open


@dataclass
class EpisodeRecorder:
    """Collects timesteps for a single rollout and serialises them.

    Usage:
        rec = EpisodeRecorder(instruction, color, action_cfg)
        # each recorded sim frame:
        rec.add(image, eef_pos, eef_quat_wxyz, gripper_closed)
        ...
        rec.set_success(True)
        rec.save(out_dir, episode_index)
    """

    instruction: str
    color: str
    action_cfg: ActionConfig
    success: bool = False
    steps: list[_Step] = field(default_factory=list)

    def add(
        self,
        image: np.ndarray,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
        gripper_closed: bool,
    ) -> None:
        """Record one observation/proprio frame."""
        img = np.ascontiguousarray(image[..., :3]).astype(np.uint8)
        gripper = self.action_cfg.gripper_closed if gripper_closed else self.action_cfg.gripper_open
        self.steps.append(
            _Step(
                image=img,
                eef_pos=np.asarray(eef_pos, dtype=np.float64).reshape(3),
                eef_rpy=quat_to_rpy(np.asarray(eef_quat_wxyz, dtype=np.float64).reshape(4)),
                gripper=float(gripper),
            )
        )

    def set_success(self, success: bool) -> None:
        self.success = bool(success)

    def __len__(self) -> int:
        return len(self.steps)

    # --- derived arrays ------------------------------------------------------
    def _states(self) -> np.ndarray:
        """[N, 7] proprio: eef xyz, rpy, gripper."""
        return np.array(
            [[*s.eef_pos, *s.eef_rpy, s.gripper] for s in self.steps],
            dtype=np.float32,
        )

    def _actions(self) -> np.ndarray:
        """[N, 7] delta-EEF actions; action[t] moves state[t] -> state[t+1].

        The last action is zero motion with the final gripper command, so it is
        a valid (if trivial) supervision target and keeps len(actions)==len(obs).
        """
        n = len(self.steps)
        acts = np.zeros((n, self.action_cfg.dim), dtype=np.float32)
        cfg = self.action_cfg
        for t in range(n - 1):
            cur, nxt = self.steps[t], self.steps[t + 1]
            dpos = np.clip(nxt.eef_pos - cur.eef_pos, -cfg.max_delta_pos, cfg.max_delta_pos)
            drot = np.clip(wrap_to_pi(nxt.eef_rpy - cur.eef_rpy), -cfg.max_delta_rot, cfg.max_delta_rot)
            acts[t, 0:3] = dpos
            acts[t, 3:6] = drot
            acts[t, 6] = nxt.gripper            # gripper is absolute, not a delta
        if n > 0:
            acts[n - 1, 6] = self.steps[-1].gripper
        return acts

    def _images(self) -> np.ndarray:
        return np.stack([s.image for s in self.steps], axis=0)

    # --- serialisation -------------------------------------------------------
    def save(self, out_dir: Path, episode_index: int) -> Path:
        """Write `episode_{index:05d}.npz` + `.json` meta to out_dir.

        Returns the path to the .npz. Raises ValueError on an empty episode.
        """
        if not self.steps:
            raise ValueError("refusing to save an empty episode")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"episode_{episode_index:05d}"
        npz_path = out_dir / f"{stem}.npz"

        images = self._images()
        states = self._states()
        actions = self._actions()

        np.savez_compressed(
            npz_path,
            images=images,
            states=states,
            actions=actions,
        )

        meta = {
            "episode_index": episode_index,
            "instruction": self.instruction,
            "color": self.color,
            "success": self.success,
            "num_steps": len(self.steps),
            "image_shape": list(images.shape[1:]),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "action_layout": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
            "state_layout": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            "gripper_convention": "1.0=closed/grasping, 0.0=open",
            "frame": "world",
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        return npz_path


def load_episode(npz_path: str | Path) -> dict:
    """Load a saved episode back into a dict of arrays + its meta sidecar."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        out = {k: data[k] for k in data.files}
    meta_path = npz_path.with_suffix(".json")
    if meta_path.exists():
        out["meta"] = json.loads(meta_path.read_text())
    return out
