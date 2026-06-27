"""vla_collect — Isaac Sim 6.0.0 data-collection pipeline for fine-tuning OpenVLA.

A 6-axis UR10 arm on a table performs a scripted pick-and-place task: place a
coloured cube on top of a red rectangle. Each rollout is recorded as an episode
of (image, proprioceptive state, 7-DoF end-effector action, language
instruction) tuples — the shape OpenVLA expects — and can be converted to an
RLDS/TFDS dataset for fine-tuning.

Submodules:
    config    — all tunables (workspace, colours, camera, action space, paths).
    scene     — builds the world: table, UR10, red pad, coloured cubes, camera.
    recorder  — pure-numpy episode buffer + on-disk format (no Isaac import).
    collect   — standalone Isaac Sim entry point that runs the rollouts.

Tools (run with the venv's plain python, no Isaac needed):
    convert_rlds.py    — recorded episodes -> OpenVLA RLDS dataset.
    inspect_dataset.py — sanity-check a recorded run.
"""

__all__ = ["config", "recorder"]
