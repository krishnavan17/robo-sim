# robo-sim

Robotics simulation environment wiring **NVIDIA Isaac Sim 6.0.0** and **ROS 2 Jazzy**
together behind a single Python 3.12 virtual environment, so that one interpreter can
import Isaac Sim (`isaacsim`, `omni`, `pxr`), ROS 2 (`rclpy`, `cv_bridge`, `*_msgs`),
and your pip packages (`numpy`, `opencv`, `ultralytics`) at the same time.

## Reference environment

This repo targets — and its `install.sh` reproduces — the following system:

| Component   | Version / detail                                                    |
|-------------|---------------------------------------------------------------------|
| OS          | Ubuntu 24.04 LTS (Noble), x86_64, GLIBC 2.39                        |
| GPU         | NVIDIA RTX 3090, proprietary driver 595.x, CUDA 13.2               |
| ROS 2       | Jazzy (`ros-jazzy-desktop` + `ros-dev-tools`), `ROS_DOMAIN_ID=42`  |
| Isaac Sim   | 6.0.0 standalone, unzipped into `$HOME`                            |
| Python      | 3.12                                                                |

Isaac Sim and ROS 2 are **not** on PyPI and are **not** pip-installed. They are installed
at the system level and then *exposed* to the venv (see [How the venv works](#how-the-venv-works)).

> Running under WSL2? See [WSL2 support](#wsl2-support) — the GPU driver step works
> differently and the scripts handle it automatically.

## Quick start

### Fresh machine (full provision)

```bash
./install.sh
```

This installs the system-level pieces (NVIDIA driver, ROS 2 Jazzy, Isaac Sim 6.0.0) and
then creates the Python venv. Stages can be skipped individually:

```bash
./install.sh --no-driver    # NVIDIA driver already present (e.g. after a reboot)
./install_gpu.sh            # install the GPU driver on its own (then reboot)
./install_gpu.sh --check    # audit the current driver/GPU, install nothing
./install.sh --no-ros       # ROS 2 already installed
./install.sh --no-isaac     # Isaac Sim already extracted
./install.sh --no-venv      # don't (re)create the venv
```

Because the Isaac Sim zip is ~13 GB, the fastest way to replicate is to copy an existing
zip to the new machine and point the installer at it:

```bash
ISAAC_ZIP=~/isaac-sim-standalone-6.0.0-linux-x86_64.zip ./install.sh
```

### Machine that already has Isaac Sim + ROS 2

If the system already matches the reference (Isaac Sim under `$HOME`, ROS 2 at
`/opt/ros/jazzy`), just build the venv:

```bash
./setup_venv.sh
source .venv/bin/activate
python -c "import numpy, cv2, ultralytics, rclpy; print('pip + ROS ok')"
```

## Running

```bash
./launch.sh                 # launch the Isaac Sim GUI
./launch.sh script.py [..]  # run a standalone Isaac Sim Python script with the venv
```

In standalone mode the script runs under the venv's `python`, with `SimulationApp`, your
pip dependencies, and ROS 2 all importable. Extra arguments are forwarded to the script.

## OpenVLA data collection (`vla_collect/`)

A self-contained pipeline that drives a **6-axis UR10 arm** on a table to perform a simple
language-conditioned task — *"place the {colour} cube on the red rectangle"* — and records
each rollout in the shape **[OpenVLA](https://openvla.github.io/)** expects for fine-tuning.

The scene spawns four coloured cubes (blue, green, yellow, red) at random positions plus a
red rectangular target pad. A scripted **RMPFlow pick-and-place controller** (NVIDIA's
bundled UR10 solver + surface gripper) executes the task while a fixed third-person camera
captures a 224×224 RGB observation each step. Because the policy is given the *colour* in
the instruction and several cubes are present, the resulting dataset teaches colour
grounding, not just "pick the only object".

### Collect

```bash
./run_collect.sh                          # 50 successful episodes -> data/raw
./run_collect.sh --num-episodes 200       # more data
./run_collect.sh --gui                     # watch in the viewport
./run_collect.sh --keep-failures           # also keep rollouts that miss the pad
```

Each episode is written as `data/raw/episode_NNNNN.npz` (+ a `.json` sidecar):

| Array     | Shape         | Meaning                                                        |
|-----------|---------------|----------------------------------------------------------------|
| `images`  | `[T,224,224,3]` uint8 | third-person RGB observation per step                 |
| `states`  | `[T,7]` float32 | proprio: end-effector `x,y,z, roll,pitch,yaw, gripper`      |
| `actions` | `[T,7]` float32 | 7-DoF action: Δ end-effector `dx..dyaw` + absolute gripper |

Gripper convention: **1.0 = closed/grasping, 0.0 = open**. By default only successful
rollouts (cube ends on the pad) are saved.

### Inspect

```bash
.venv/bin/python tools/inspect_dataset.py data/raw                       # summary
.venv/bin/python tools/inspect_dataset.py data/raw --contact-sheet s.png  # eyeball frames
./launch.sh tools/preview_camera.py --out /tmp/preview.png                # single frame (framing check)
```

### Convert to RLDS for OpenVLA

OpenVLA fine-tunes on RLDS/TFDS datasets. TensorFlow is heavy and not in
`requirements.txt`, so install it only when converting:

```bash
.venv/bin/pip install "tensorflow-cpu>=2.15" "tensorflow-datasets>=4.9"
.venv/bin/python tools/convert_rlds.py --data-dir data/raw --out-dir data/rlds
```

This writes `data/rlds/robo_pickplace/1.0.0/`, loadable in the OpenVLA fine-tune pipeline
via `tfds.builder_from_directory(...)`. The RLDS steps carry `observation.image`,
`observation.state`, `action`, and `language_instruction` — the fields OpenVLA reads.

### Layout

| Path                        | Purpose                                                          |
|-----------------------------|------------------------------------------------------------------|
| `vla_collect/config.py`     | All tunables: cubes, workspace, camera, action space, paths.     |
| `vla_collect/scene.py`      | Builds the world (table, UR10, pad, cubes, camera, lighting).    |
| `vla_collect/recorder.py`   | Pure-numpy episode buffer + on-disk `.npz`/`.json` format.       |
| `vla_collect/collect.py`    | Standalone Isaac Sim entry point that runs the scripted rollouts.|
| `tools/inspect_dataset.py`  | Summary + contact sheet of a recorded run.                       |
| `tools/convert_rlds.py`     | Recorded episodes → RLDS/TFDS for OpenVLA.                        |
| `tools/preview_camera.py`   | Dump one observation frame to check camera framing/lighting.     |
| `run_collect.sh`            | Wrapper over `launch.sh` for the collection script.              |

### Rendering notes (lessons baked into the code)

Getting clean synthetic frames out of Isaac Sim 6.0.0 took some tuning, all handled
automatically by the code:

- **Render mode** is set to `RaytracedLighting` at `SimulationApp` construction. The
  default `RealTimePathTracing` accumulates samples across frames, so per-step captures of
  a moving scene look smeared. (It can't be changed via carb settings afterwards.)
- **Lighting** — the UR10 asset ships a ~9 000 000-intensity light on its end-effector and
  the default ground plane a strong sphere light; both blow the close-up camera out to pure
  white. `scene.tame_scene_lights()` scales every light down, then a soft dome adds fill.
- **Depth of field** is disabled (`fStop = 0`) for a sharp pinhole image, and the camera
  renders at 512² then downsamples to 224² (rendering directly at 224 trips DLSS's blur).

## Scripts

| File              | Purpose                                                                       |
|-------------------|-------------------------------------------------------------------------------|
| `install.sh`      | Provision a fresh machine to match the reference environment (GPU driver, ROS 2, Isaac Sim, venv). Idempotent. |
| `install_gpu.sh`  | NVIDIA GPU driver install only — called by `install.sh`, or run standalone (`--check` to audit). WSL-aware: verifies passthrough instead of installing on WSL. |
| `setup_venv.sh`   | Create the `.venv`, install `requirements.txt`, and wire Isaac Sim + ROS 2 into it. |
| `launch.sh`       | Activate the venv and start Isaac Sim (GUI) or run a standalone Python script. |
| `run_collect.sh`  | Run the OpenVLA pick-and-place data collection (wrapper over `launch.sh`). See [OpenVLA data collection](#openvla-data-collection-vla_collect). |
| `requirements.txt`| Pip dependencies only (numpy, opencv-python, ultralytics, pillow). **Not** Isaac/ROS. |

## How the venv works

`setup_venv.sh` builds an isolated Python 3.12 venv and bridges the two external installs:

1. **`.pth` file** — adds Isaac Sim's `kit/python` site-packages and
   `exts/isaacsim.simulation_app` to `sys.path`.
2. **Activate hook** — appends `source` lines to `.venv/bin/activate` for
   `/opt/ros/jazzy/setup.bash` and Isaac Sim's `setup_python_env.sh` /
   `setup_ros_env.sh`, which set the `PYTHONPATH` and `LD_LIBRARY_PATH` (native `.so`
   libraries) that a `.pth` file alone cannot provide.

> **Do not** add `isaacsim`, `omni`, `rclpy`, or `ros2` packages to `requirements.txt` —
> they would fail to build or shadow the working system installs.

## Configuration

All scripts honor these environment variables (with sensible defaults):

| Variable         | Default                      | Used by                  |
|------------------|------------------------------|--------------------------|
| `ISAAC_SIM_ROOT` | `$HOME`                      | all                      |
| `ROS_SETUP`      | `/opt/ros/jazzy/setup.bash`  | `setup_venv.sh`          |
| `VENV_DIR`       | `./.venv`                    | `setup_venv.sh`, `launch.sh` |
| `ROS_DISTRO`     | `jazzy`                      | `install.sh`             |
| `NVIDIA_DRIVER`  | `595`                        | `install.sh`             |
| `ROS_DOMAIN_ID`  | `42`                         | `install.sh`             |
| `ISAAC_ZIP`      | *(download)*                 | `install.sh`             |
| `ISAAC_ZIP_URL`  | NVIDIA download URL          | `install.sh`             |

## WSL2 support

The scripts run under **WSL2** (Ubuntu 24.04) with one important difference: the GPU.

Under WSL you must **not** install a Linux NVIDIA driver. The GPU is provided by the
**Windows host driver** and exposed to the distro via `/dev/dxg`; installing a Linux
driver inside WSL breaks that passthrough. Both `install.sh` and `install_gpu.sh` detect
WSL (via `/proc/sys/kernel/osrelease`) and automatically switch the driver stage to a
*verify-only* path — they check that `nvidia-smi` passthrough works instead of running
`apt install nvidia-driver-*`.

**Setup on WSL2:**

1. Install the latest NVIDIA driver on **Windows** (it includes WSL GPU support). Nothing
   to install inside WSL for the GPU.
2. Ensure you are on **WSL2** (not WSL1) with a recent kernel.
3. Run the installer normally — the driver stage will just verify passthrough:

   ```bash
   ./install.sh                 # GPU stage auto-detects WSL and only verifies nvidia-smi
   ./install_gpu.sh --check     # confirm GPU passthrough on its own
   ```

| Piece                      | WSL2 status                                                        |
|----------------------------|--------------------------------------------------------------------|
| Bash scripts, ROS 2, venv  | ✅ Work unchanged                                                   |
| GPU driver                 | ⚠️ Installed on **Windows**, not in WSL (scripts verify-only)       |
| Isaac Sim compute / RTX    | ✅ Works on WSL2 with a recent Windows driver                       |
| Isaac Sim **GUI**          | ⚠️ Flaky through WSLg — prefer headless **streaming** or standalone Python |

On WSL, prefer headless usage — `isaac-sim.streaming.sh` or `./launch.sh script.py` — over
the GUI (`./launch.sh`), since RTX rendering through WSLg is unreliable.

## Notes

- The NVIDIA driver install requires a **reboot** before the GPU is usable; re-run
  `./install.sh --no-driver` afterwards to continue. *(Native Linux only — on WSL the
  driver lives on the Windows host; see [WSL2 support](#wsl2-support).)*
- Isaac Sim modules (`isaacsim`, `omni`, `pxr`) require a GPU/display session to import
  fully; `numpy`/`cv2`/`ultralytics`/`rclpy` can be verified without one.
- The Isaac Sim download URL may be gated or change; if the download fails, fetch the zip
  from <https://developer.nvidia.com/isaac/sim> and pass it via `ISAAC_ZIP=`.
