#!/usr/bin/env python3
"""
============================================================
Perry's Wan 2.2 Pipeline — Setup Verification Script
============================================================
Run this before your first generation to confirm everything
is in place. Prints a clear pass/fail for each requirement.

USAGE:
    python3 scripts/verify_setup.py

EXIT CODES:
    0 — all checks passed
    1 — one or more critical checks failed
============================================================
"""

import sys
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
COMFYUI_DIR = BASE_DIR / "ComfyUI"
MODELS_DIR  = BASE_DIR / "models"

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "
INFO = "  ℹ️ "

failures = []
warnings = []


def check(label: str, condition: bool, fail_msg: str, warn_only: bool = False):
    if condition:
        print(f"{PASS}  {label}")
    else:
        marker = WARN if warn_only else FAIL
        print(f"{marker}  {label}")
        print(f"         → {fail_msg}")
        if warn_only:
            warnings.append(label)
        else:
            failures.append(label)


def section(title: str):
    print()
    print(f"  {'─' * 55}")
    print(f"  {title}")
    print(f"  {'─' * 55}")


# ── 1. Python version ─────────────────────────────────────────
section("Python")
import platform
py_ver = tuple(int(x) for x in platform.python_version_tuple()[:2])
check(
    f"Python version: {platform.python_version()}",
    py_ver >= (3, 10),
    "Python 3.10+ is required. Install with: sudo apt install python3.10",
)

# ── 2. GPU / CUDA ─────────────────────────────────────────────
section("GPU & CUDA")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    check("PyTorch installed", True, "")
    check(
        f"CUDA available (torch sees GPU)",
        cuda_ok,
        "CUDA not available. Verify NVIDIA drivers and PyTorch CUDA build.",
    )
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check(f"GPU detected: {gpu_name}", True, "")
        check(
            f"VRAM: {vram_gb:.1f} GB",
            vram_gb >= 20,
            "Need at least 20 GB VRAM for the 14B Wan model. "
            "A100-80GB (your card) has 80 GB — this should pass.",
        )
        check(
            f"PyTorch CUDA build: {torch.version.cuda}",
            torch.version.cuda is not None,
            "PyTorch was installed without CUDA support. "
            "Reinstall: pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124",
        )
except ImportError:
    check("PyTorch installed", False,
          "Install: pip install torch==2.5.1+cu124 torchvision torchaudio "
          "--index-url https://download.pytorch.org/whl/cu124")

# Check nvidia-smi
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        check(f"nvidia-smi: {parts[0]} | Driver {parts[1]} | {parts[2]}", True, "")
    else:
        check("nvidia-smi accessible", False, "nvidia-smi command failed")
except FileNotFoundError:
    check("nvidia-smi accessible", False,
          "nvidia-smi not found. Install NVIDIA drivers.")

# ── 3. ComfyUI installation ───────────────────────────────────
section("ComfyUI Installation")
check(
    f"ComfyUI directory exists: {COMFYUI_DIR}",
    COMFYUI_DIR.exists(),
    f"Run: bash scripts/setup_comfyui.sh",
)
check(
    "ComfyUI main.py present",
    (COMFYUI_DIR / "main.py").exists(),
    "ComfyUI is incomplete. Re-run: bash scripts/setup_comfyui.sh",
)
check(
    "extra_model_paths.yaml configured",
    (COMFYUI_DIR / "extra_model_paths.yaml").exists(),
    "Run: bash scripts/setup_comfyui.sh (regenerates the config)",
)

# ── 4. Custom Nodes ───────────────────────────────────────────
section("Custom Nodes")
CN_DIR = COMFYUI_DIR / "custom_nodes"

def node_installed(folder_name: str) -> bool:
    return (CN_DIR / folder_name / "__init__.py").exists() or \
           (CN_DIR / folder_name).is_dir()

check(
    "KJNodes installed (ComfyUI-KJNodes)",
    node_installed("ComfyUI-KJNodes"),
    "Run: cd ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-KJNodes",
)
check(
    "VideoHelperSuite installed",
    node_installed("ComfyUI-VideoHelperSuite"),
    "Run: cd ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
)
check(
    "Comfyroll installed",
    node_installed("ComfyUI_Comfyroll_CustomNodes"),
    "Run: cd ComfyUI/custom_nodes && git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes",
)

# Ask_Gemini / MasterKey — search across all custom node files
ask_gemini_found = False
if CN_DIR.exists():
    for py_file in CN_DIR.rglob("*.py"):
        try:
            text = py_file.read_text(errors="ignore")
            if "Ask_Gemini" in text and "MasterKey" in text:
                ask_gemini_found = True
                print(f"{PASS}  Ask_Gemini/MasterKey found in: {py_file.parent.name}")
                break
            elif "Ask_Gemini" in text or "MasterKey" in text:
                ask_gemini_found = True
                print(f"{WARN}  Ask_Gemini OR MasterKey found in: {py_file.parent.name} (partial)")
                break
        except Exception:
            continue

if not ask_gemini_found:
    check(
        "Ask_Gemini / MasterKey node installed",
        False,
        "Install via ComfyUI Manager → Install Missing Nodes, or find the\n"
        "         source repo with: grep -r 'class Ask_Gemini' ComfyUI/custom_nodes/",
    )

# ── 5. Required model files ───────────────────────────────────
section("Model Files")

REQUIRED_MODELS = [
    (MODELS_DIR / "vae"              / "wan_2.1_vae.safetensors",
     "VAE",
     "Run: python3 scripts/download_models.py"),

    (MODELS_DIR / "text_encoders"    / "umt5_xxl_fp16.safetensors",
     "Text Encoder (UMT5-XXL fp16)",
     "Run: python3 scripts/download_models.py"),

    (MODELS_DIR / "diffusion_models" / "smoothMixWan2214BI2V_i2vHigh.safetensors",
     "Base Model HIGH",
     "Download from CivitAI: search 'smoothMixWan2214BI2V'"),

    (MODELS_DIR / "diffusion_models" / "smoothMixWan2214BI2V_i2vLow.safetensors",
     "Base Model LOW",
     "Download from CivitAI: search 'smoothMixWan2214BI2V'"),

    (MODELS_DIR / "loras"            / "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
     "SVI Pro LoRA HIGH",
     "Download from CivitAI: search 'SVI_v2_PRO_Wan2.2'"),

    (MODELS_DIR / "loras"            / "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors",
     "SVI Pro LoRA LOW",
     "Download from CivitAI: search 'SVI_v2_PRO_Wan2.2'"),
]

for path, label, fix_msg in REQUIRED_MODELS:
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"{PASS}  {label:<35} ({size_mb:.0f} MB)")
    else:
        check(label, False, fix_msg)

# ── 6. Workflow files ─────────────────────────────────────────
section("Workflow JSON Files")

WORKFLOWS = {
    "jewellery_workflow.json": "Jewellery / Gemstone / Watch videos",
    "tech_workflow.json":      "Tech product videos (iPhone, iPad, laptop...)",
    "close_workflow.json":     "Close-up hand/physics videos",
}

workflow_dir = BASE_DIR / "workflows"
for fname, desc in WORKFLOWS.items():
    path = workflow_dir / fname
    check(
        f"{desc}: {fname}",
        path.exists(),
        f"Place the workflow JSON file at: {path}",
    )

# ── 7. Environment / API key ──────────────────────────────────
section("Environment Variables")

env_file = BASE_DIR / ".env"
check(
    ".env file created from .env.example",
    env_file.exists(),
    "Run: cp .env.example .env  then fill in GEMINI_API_KEY",
    warn_only=True,
)

# Load .env if exists
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

gemini_key = os.environ.get("GEMINI_API_KEY", "")
check(
    "GEMINI_API_KEY is set",
    bool(gemini_key) and gemini_key != "AIzaSy_PASTE_YOUR_KEY_HERE",
    "Set GEMINI_API_KEY in .env file. Get key at: https://aistudio.google.com/app/apikey",
)

# ── 8. Key Python packages ────────────────────────────────────
section("Key Python Packages")

def pkg_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None

for pkg in ["transformers", "safetensors", "huggingface_hub",
            "google-generativeai", "av", "pillow"]:
    ver = pkg_version(pkg)
    check(
        f"{pkg}: {ver}" if ver else pkg,
        ver is not None,
        f"Install: pip install {pkg}",
    )

# ── Final summary ─────────────────────────────────────────────
print()
print("  " + "═" * 55)
if not failures:
    if not warnings:
        print(f"  🎉 ALL CHECKS PASSED — Pipeline is ready to run!")
    else:
        print(f"  ✅ PASSED with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"      ⚠️   {w}")
    print()
    print("  START COMFYUI:")
    print("    Bare metal : bash scripts/start_comfyui.sh")
    print("    Docker     : bash scripts/run_docker.sh")
    print("    Compose    : docker compose up")
    print()
    print("  Then open: http://localhost:8188")
else:
    print(f"  ❌ {len(failures)} CRITICAL FAILURE(S) — fix before running:")
    for f in failures:
        print(f"      ✗  {f}")
    if warnings:
        print(f"\n  ⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"      !  {w}")
print("  " + "═" * 55)
print()

sys.exit(0 if not failures else 1)
