#!/usr/bin/env bash
# =============================================================================
# install.sh — provision a machine to match the robo-sim reference environment
# =============================================================================
# Reference system this script replicates:
#   * Ubuntu 24.04 LTS (Noble), x86_64, GLIBC 2.39
#   * NVIDIA GPU (reference: RTX 3090) + proprietary driver (reference: 595.x)
#   * ROS 2 Jazzy   -> /opt/ros/jazzy  (ros-jazzy-desktop + ros-dev-tools)
#       - ROS_DOMAIN_ID=42, sourced from ~/.bashrc
#   * NVIDIA Isaac Sim 6.0.0 (standalone) -> unzipped into $HOME
#       - provides isaac-sim.sh, setup_python_env.sh, setup_ros_env.sh, kit/, exts/
#   * Python 3.12 venv (created separately by ./setup_venv.sh)
#
# This script installs the SYSTEM-level pieces (driver, ROS 2, Isaac Sim).
# The Python venv that bridges pip + Isaac + ROS is created afterwards by
# ./setup_venv.sh (run it once this script finishes).
#
# Usage:
#   ./install.sh                 # full install (driver + ROS 2 + Isaac Sim + venv)
#   ./install.sh --no-driver     # skip the NVIDIA driver step (already installed)
#   ./install.sh --no-isaac      # skip Isaac Sim download/extract
#   ./install.sh --no-venv       # skip the final ./setup_venv.sh call
#
# Override defaults via env:
#   ISAAC_SIM_ROOT   install dir for Isaac Sim        (default: $HOME)
#   ISAAC_ZIP        path to a pre-downloaded zip      (default: download it)
#   ISAAC_ZIP_URL    download URL for the standalone zip
#   ROS_DISTRO       ROS 2 distro                      (default: jazzy)
#   NVIDIA_DRIVER    driver package version to apt-install (default: 595)
#   ROS_DOMAIN_ID    DDS domain id baked into ~/.bashrc (default: 42)
# =============================================================================
set -euo pipefail

# --- tunables ----------------------------------------------------------------
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
NVIDIA_DRIVER="${NVIDIA_DRIVER:-595}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ISAAC_VERSION="6.0.0"
ISAAC_ZIP="${ISAAC_ZIP:-}"
ISAAC_ZIP_URL="${ISAAC_ZIP_URL:-https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-${ISAAC_VERSION}-linux-x86_64.zip}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DO_DRIVER=1; DO_ROS=1; DO_ISAAC=1; DO_VENV=1
for arg in "$@"; do
    case "$arg" in
        --no-driver) DO_DRIVER=0 ;;
        --no-ros)    DO_ROS=0 ;;
        --no-isaac)  DO_ISAAC=0 ;;
        --no-venv)   DO_VENV=0 ;;
        -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

log()  { printf '\n\033[1;36m>> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 0. sanity ---------------------------------------------------------------
log "Checking host compatibility"
[ "$(uname -m)" = "x86_64" ] || die "This script targets x86_64 (found $(uname -m))."
if [ -f /etc/os-release ]; then . /etc/os-release; fi
[ "${VERSION_ID:-}" = "24.04" ] || warn "Reference OS is Ubuntu 24.04; found '${VERSION_ID:-unknown}'. Continuing."
[ "$(id -u)" -ne 0 ] || die "Run as a normal user (the script uses sudo where needed), not root."
command -v sudo >/dev/null || die "sudo is required."

# =============================================================================
# 1. NVIDIA driver
# =============================================================================
if [ "$DO_DRIVER" -eq 1 ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        log "NVIDIA driver already present:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    else
        log "Installing NVIDIA driver (nvidia-driver-${NVIDIA_DRIVER})"
        sudo apt-get update
        sudo apt-get install -y "nvidia-driver-${NVIDIA_DRIVER}" || \
            warn "Driver package nvidia-driver-${NVIDIA_DRIVER} not found; try 'ubuntu-drivers devices' to pick one."
        warn "A REBOOT is required to load the new NVIDIA driver. Re-run with --no-driver afterwards."
    fi
else
    log "Skipping NVIDIA driver (--no-driver)"
fi

# =============================================================================
# 2. ROS 2 Jazzy  (mirrors /opt/ros/jazzy on the reference system)
# =============================================================================
if [ "$DO_ROS" -eq 1 ]; then
    if [ -d "/opt/ros/${ROS_DISTRO}" ]; then
        log "ROS 2 ${ROS_DISTRO} already installed at /opt/ros/${ROS_DISTRO}"
    else
        log "Installing ROS 2 ${ROS_DISTRO} (desktop) + dev tools"
        # Locale (ROS requires a UTF-8 locale)
        sudo apt-get update && sudo apt-get install -y locales
        sudo locale-gen en_US en_US.UTF-8
        sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

        # Enable the Ubuntu Universe repo
        sudo apt-get install -y software-properties-common curl
        sudo add-apt-repository -y universe

        # Add the ROS 2 apt repository via the ros-apt-source package (modern method)
        ROS_APT_DEB="$(mktemp --suffix=.deb)"
        ROS_APT_VER="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
                        | grep -oP '"tag_name":\s*"\K[^"]+' || echo '')"
        [ -n "$ROS_APT_VER" ] || die "Could not determine latest ros-apt-source version."
        curl -fsSL -o "$ROS_APT_DEB" \
            "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_VER}/ros2-apt-source_${ROS_APT_VER}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
        sudo apt-get install -y "$ROS_APT_DEB"
        rm -f "$ROS_APT_DEB"

        sudo apt-get update && sudo apt-get upgrade -y
        sudo apt-get install -y \
            "ros-${ROS_DISTRO}-desktop" \
            ros-dev-tools \
            python3-rosdep \
            python3-colcon-common-extensions

        # rosdep init (idempotent)
        if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
            sudo rosdep init || true
        fi
        rosdep update || true
    fi

    # --- wire ROS into ~/.bashrc (idempotent) --------------------------------
    BRC="$HOME/.bashrc"
    if ! grep -q "source /opt/ros/${ROS_DISTRO}/setup.bash" "$BRC" 2>/dev/null; then
        log "Adding ROS 2 source line to ~/.bashrc"
        {
            echo ""
            echo "# ---- robo-sim: ROS 2 ${ROS_DISTRO} (added by install.sh) ----"
            echo "source /opt/ros/${ROS_DISTRO}/setup.bash"
            echo "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
        } >> "$BRC"
    else
        log "ROS 2 already sourced in ~/.bashrc"
    fi
else
    log "Skipping ROS 2 (--no-ros)"
fi

# =============================================================================
# 3. Isaac Sim 6.0.0 (standalone, unzipped into $ISAAC_SIM_ROOT)
# =============================================================================
if [ "$DO_ISAAC" -eq 1 ]; then
    if [ -f "$ISAAC_SIM_ROOT/isaac-sim.sh" ] && [ -d "$ISAAC_SIM_ROOT/kit" ]; then
        log "Isaac Sim already present at $ISAAC_SIM_ROOT"
        [ -f "$ISAAC_SIM_ROOT/VERSION" ] && cat "$ISAAC_SIM_ROOT/VERSION" && echo
    else
        log "Installing Isaac Sim ${ISAAC_VERSION} into $ISAAC_SIM_ROOT"
        sudo apt-get update && sudo apt-get install -y unzip curl

        if [ -z "$ISAAC_ZIP" ]; then
            ISAAC_ZIP="$ISAAC_SIM_ROOT/isaac-sim-standalone-${ISAAC_VERSION}-linux-x86_64.zip"
            if [ ! -f "$ISAAC_ZIP" ]; then
                log "Downloading Isaac Sim zip (~13 GB) from:"
                echo "   $ISAAC_ZIP_URL"
                curl -fL --retry 3 -o "$ISAAC_ZIP" "$ISAAC_ZIP_URL" || die \
                    "Download failed. Grab the zip manually from https://developer.nvidia.com/isaac/sim and set ISAAC_ZIP=/path/to/zip."
            else
                log "Using already-downloaded zip: $ISAAC_ZIP"
            fi
        fi

        [ -f "$ISAAC_ZIP" ] || die "Isaac Sim zip not found: $ISAAC_ZIP"
        log "Extracting $ISAAC_ZIP -> $ISAAC_SIM_ROOT"
        mkdir -p "$ISAAC_SIM_ROOT"
        unzip -q -o "$ISAAC_ZIP" -d "$ISAAC_SIM_ROOT"

        [ -f "$ISAAC_SIM_ROOT/setup_python_env.sh" ] || die \
            "Extraction did not produce setup_python_env.sh in $ISAAC_SIM_ROOT — check the zip contents."

        # First-run compatibility check (non-fatal)
        if [ -x "$ISAAC_SIM_ROOT/isaac-sim.compatibility_check.sh" ]; then
            log "Running Isaac Sim compatibility check"
            "$ISAAC_SIM_ROOT/isaac-sim.compatibility_check.sh" || \
                warn "Compatibility check reported issues — review the output above."
        fi
    fi
else
    log "Skipping Isaac Sim (--no-isaac)"
fi

# =============================================================================
# 4. Python venv (delegates to setup_venv.sh)
# =============================================================================
if [ "$DO_VENV" -eq 1 ]; then
    if [ -x "$REPO_DIR/setup_venv.sh" ]; then
        log "Creating the robo-sim Python venv (./setup_venv.sh)"
        sudo apt-get install -y python3.12 python3.12-venv python3-pip || true
        ISAAC_SIM_ROOT="$ISAAC_SIM_ROOT" ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash" \
            "$REPO_DIR/setup_venv.sh"
    else
        warn "setup_venv.sh not found/executable in $REPO_DIR — skipping venv step."
    fi
else
    log "Skipping venv (--no-venv)"
fi

# =============================================================================
# Done
# =============================================================================
cat <<EOF

==============================================================
 robo-sim environment install complete.

 Installed (to match the reference system):
   * ROS 2 ${ROS_DISTRO}            -> /opt/ros/${ROS_DISTRO}  (ROS_DOMAIN_ID=${ROS_DOMAIN_ID})
   * Isaac Sim ${ISAAC_VERSION}        -> ${ISAAC_SIM_ROOT}
   * Python venv          -> ${REPO_DIR}/.venv

 Next steps:
   1. If the NVIDIA driver was just installed: reboot.
   2. Open a new shell (so ~/.bashrc re-sources ROS 2), then:
        source ${REPO_DIR}/.venv/bin/activate
        python -c "import numpy, cv2, ultralytics, rclpy; print('pip + ROS ok')"
   3. Launch Isaac Sim:
        ${REPO_DIR}/launch.sh            # GUI
        ${REPO_DIR}/launch.sh script.py  # standalone python
==============================================================
EOF
