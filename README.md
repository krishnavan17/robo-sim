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

## Quick start

### Fresh machine (full provision)

```bash
./install.sh
```

This installs the system-level pieces (NVIDIA driver, ROS 2 Jazzy, Isaac Sim 6.0.0) and
then creates the Python venv. Stages can be skipped individually:

```bash
./install.sh --no-driver    # NVIDIA driver already present (e.g. after a reboot)
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

## Scripts

| File              | Purpose                                                                       |
|-------------------|-------------------------------------------------------------------------------|
| `install.sh`      | Provision a fresh machine to match the reference environment (driver, ROS 2, Isaac Sim, venv). Idempotent. |
| `setup_venv.sh`   | Create the `.venv`, install `requirements.txt`, and wire Isaac Sim + ROS 2 into it. |
| `launch.sh`       | Activate the venv and start Isaac Sim (GUI) or run a standalone Python script. |
| `requirements.txt`| Pip dependencies only (numpy, opencv-python, ultralytics). **Not** Isaac/ROS. |

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

## Notes

- The NVIDIA driver install requires a **reboot** before the GPU is usable; re-run
  `./install.sh --no-driver` afterwards to continue.
- Isaac Sim modules (`isaacsim`, `omni`, `pxr`) require a GPU/display session to import
  fully; `numpy`/`cv2`/`ultralytics`/`rclpy` can be verified without one.
- The Isaac Sim download URL may be gated or change; if the download fails, fetch the zip
  from <https://developer.nvidia.com/isaac/sim> and pass it via `ISAAC_ZIP=`.
