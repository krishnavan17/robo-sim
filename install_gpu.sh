#!/usr/bin/env bash
# =============================================================================
# install_gpu.sh — install the NVIDIA GPU driver for the robo-sim environment
# =============================================================================
# Reference system this replicates:
#   * NVIDIA RTX 3090 + proprietary driver 595.x  (CUDA 13.2 runtime)
#
# Split out of install.sh so the (reboot-requiring) driver step can be run,
# skipped, or re-run independently. install.sh calls this script for its
# driver stage unless invoked with --no-driver.
#
# Usage:
#   ./install_gpu.sh            # install nvidia-driver-$NVIDIA_DRIVER
#   ./install_gpu.sh --check    # report the current driver/GPU, install nothing
#
# Override defaults via env:
#   NVIDIA_DRIVER    driver package version to apt-install (default: 595)
# =============================================================================
set -euo pipefail

NVIDIA_DRIVER="${NVIDIA_DRIVER:-595}"

log()  { printf '\n\033[1;36m>> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check)   CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# --- sanity ------------------------------------------------------------------
[ "$(uname -m)" = "x86_64" ] || die "This script targets x86_64 (found $(uname -m))."
[ "$(id -u)" -ne 0 ] || die "Run as a normal user (the script uses sudo where needed), not root."
command -v sudo >/dev/null || die "sudo is required."

# --- WSL2 guard --------------------------------------------------------------
# Under WSL the GPU is provided by the *Windows* NVIDIA driver and exposed to
# the distro via /dev/dxg. Installing a Linux driver here is wrong and can
# break that passthrough, so we refuse and verify the passthrough instead.
is_wsl() { grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease 2>/dev/null; }
if is_wsl; then
    warn "WSL detected — do NOT install a Linux NVIDIA driver inside WSL."
    cat >&2 <<'WSLMSG'
   On WSL2 the GPU comes from the Windows host driver:
     1. Install the latest NVIDIA driver on WINDOWS (it includes WSL support).
     2. Use WSL2 (not WSL1) with a recent kernel.
     3. nvidia-smi then works inside WSL with no Linux driver installed.
WSLMSG
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        log "GPU passthrough is working:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
        exit 0
    fi
    die "No GPU passthrough yet — install the Windows NVIDIA driver, then re-run with --check."
fi

# --- already installed? ------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver already present:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    exit 0
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "No working NVIDIA driver detected (nvidia-smi unavailable)."
    command -v ubuntu-drivers >/dev/null 2>&1 && ubuntu-drivers devices || true
    exit 1
fi

# --- install -----------------------------------------------------------------
log "Installing NVIDIA driver (nvidia-driver-${NVIDIA_DRIVER})"
sudo apt-get update
sudo apt-get install -y "nvidia-driver-${NVIDIA_DRIVER}" || \
    warn "Driver package nvidia-driver-${NVIDIA_DRIVER} not found; try 'ubuntu-drivers devices' to pick one."
warn "A REBOOT is required to load the new NVIDIA driver, then re-run install.sh --no-driver."
