#!/usr/bin/env bash
# =============================================================================
# run_collect.sh — collect OpenVLA fine-tuning data with the UR10 pick-place sim
# =============================================================================
# Thin wrapper over ./launch.sh that runs the standalone collection script under
# the venv + Isaac Sim. All arguments are forwarded to vla_collect/collect.py,
# so any of its flags work:
#
#   ./run_collect.sh                              # defaults (50 episodes -> data/raw)
#   ./run_collect.sh --num-episodes 200           # more data
#   ./run_collect.sh --data-dir data/raw --seed 1 # explicit output + seed
#   ./run_collect.sh --gui                         # watch it in the viewport
#   ./run_collect.sh --keep-failures               # also save unsuccessful rollouts
#
# After collecting, inspect and convert:
#   .venv/bin/python tools/inspect_dataset.py data/raw
#   .venv/bin/python tools/convert_rlds.py --data-dir data/raw --out-dir data/rlds
# =============================================================================
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$REPO_DIR/launch.sh" "$REPO_DIR/vla_collect/collect.py" "$@"
