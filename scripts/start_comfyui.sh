#!/usr/bin/env bash
# ============================================================
# Perry's Wan 2.2 Pipeline — Start ComfyUI (Bare Metal / In-Container)
# ============================================================
# Run this to launch the ComfyUI web server.
# Open http://localhost:8188 in your browser after it starts.
#
# USAGE:
#   bash scripts/start_comfyui.sh
#   bash scripts/start_comfyui.sh --lowvram      # if VRAM < 24 GB
#   bash scripts/start_comfyui.sh --port 8189    # custom port
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
COMFYUI_DIR="${BASE_DIR}/ComfyUI"

# Load .env if it exists
if [[ -f "${BASE_DIR}/.env" ]]; then
    set -a
    source "${BASE_DIR}/.env"
    set +a
fi

PORT="${COMFYUI_PORT:-8188}"

# ── GPU check ─────────────────────────────────────────────────
echo
echo "══════════════════════════════════════════════════"
echo "  Perry's Wan 2.2 Pipeline — Starting ComfyUI"
echo "══════════════════════════════════════════════════"
echo
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU    : %s\n  VRAM   : %s MiB total, %s MiB free\n  Driver : %s\n", $1, $2, $3, $4}'
echo

# ── Verify ComfyUI exists ─────────────────────────────────────
if [[ ! -f "${COMFYUI_DIR}/main.py" ]]; then
    echo "[ERROR] ComfyUI not found at ${COMFYUI_DIR}"
    echo "        Run setup first: bash scripts/setup_comfyui.sh"
    exit 1
fi

# ── Build extra args from command line ────────────────────────
EXTRA_ARGS="$*"

# Detect available VRAM and suggest flags
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "$VRAM_MB" -ge 40000 ]]; then
    VRAM_FLAG=""   # A100 80GB — no flag needed, runs full precision
    echo "  ✓ A100 detected (${VRAM_MB} MiB VRAM) — running without memory flags"
elif [[ "$VRAM_MB" -ge 24000 ]]; then
    VRAM_FLAG=""
    echo "  ✓ High-VRAM GPU (${VRAM_MB} MiB) — running without memory flags"
elif [[ "$VRAM_MB" -ge 16000 ]]; then
    VRAM_FLAG="--lowvram"
    echo "  ! Medium VRAM (${VRAM_MB} MiB) — using --lowvram"
else
    VRAM_FLAG="--lowvram --cpu-vae"
    echo "  ! Low VRAM (${VRAM_MB} MiB) — using --lowvram --cpu-vae"
fi

echo
echo "  Access URL : http://localhost:${PORT}"
echo "  Workflows  : ${COMFYUI_DIR}/workflows/"
echo "  Models     : ${BASE_DIR}/models/"
echo "  Outputs    : ${BASE_DIR}/outputs/"
echo
echo "  Press Ctrl+C to stop."
echo "══════════════════════════════════════════════════"
echo

cd "${COMFYUI_DIR}"
python3 main.py \
    --listen 0.0.0.0 \
    --port "${PORT}" \
    --enable-cors-header \
    --preview-method auto \
    ${VRAM_FLAG} \
    ${EXTRA_ARGS}
