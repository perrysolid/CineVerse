#!/usr/bin/env python3
"""
============================================================
Perry's Wan 2.2 Pipeline — Model Download Script
============================================================
Downloads all required models for the pipeline.

USAGE:
    python scripts/download_models.py

WHAT IT DOWNLOADS (automatically, from HuggingFace):
    ✅  wan_2.1_vae.safetensors         → models/vae/
    ✅  umt5_xxl_fp16.safetensors       → models/text_encoders/

WHAT YOU MUST DOWNLOAD MANUALLY (from CivitAI):
    ⚠️  smoothMixWan2214BI2V_i2vHigh.safetensors  → models/diffusion_models/
    ⚠️  smoothMixWan2214BI2V_i2vLow.safetensors   → models/diffusion_models/
    ⚠️  SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors → models/loras/
    ⚠️  SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors  → models/loras/

PREREQUISITES:
    pip install huggingface_hub requests tqdm
============================================================
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from tqdm import tqdm
    import requests
except ImportError:
    print("[ERROR] Missing dependencies. Run:")
    print("        pip install huggingface_hub requests tqdm")
    sys.exit(1)

# ── Directory layout ─────────────────────────────────────────────────────────
# Resolve base path: script lives in perry-wan-pipeline/scripts/
# so base = parent of scripts/
SCRIPT_DIR   = Path(__file__).resolve().parent
BASE_DIR     = SCRIPT_DIR.parent
MODELS_DIR   = BASE_DIR / "models"

DIRS = {
    "diffusion_models": MODELS_DIR / "diffusion_models",
    "vae":              MODELS_DIR / "vae",
    "text_encoders":    MODELS_DIR / "text_encoders",
    "loras":            MODELS_DIR / "loras",
    "clip":             MODELS_DIR / "clip",
}

# ── HuggingFace models (auto-downloadable) ───────────────────────────────────
HF_MODELS = [
    {
        "description": "Wan 2.1 VAE",
        "repo_id":     "Wan-AI/Wan2.1-I2V-14B-480P",
        "filename":    "wan_2.1_vae.safetensors",
        "dest_dir":    "vae",
        "required":    True,
    },
    {
        "description": "UMT5-XXL Text Encoder (fp16)",
        "repo_id":     "Wan-AI/Wan2.1-I2V-14B-480P",
        "filename":    "text_encoder/umt5_xxl_fp16.safetensors",
        "dest_dir":    "text_encoders",
        "rename_to":   "umt5_xxl_fp16.safetensors",
        "required":    True,
    },
]

# ── CivitAI models (manual download required) ────────────────────────────────
CIVITAI_MODELS = [
    {
        "description": "smoothMixWan2214BI2V — i2vHigh (base model HIGH pass)",
        "filename":    "smoothMixWan2214BI2V_i2vHigh.safetensors",
        "dest_dir":    "diffusion_models",
        "search_url":  "https://civitai.com/search/models?query=smoothMixWan2214BI2V",
        "notes":       "Download the _i2vHigh variant",
        "size_gb":     ~27,
    },
    {
        "description": "smoothMixWan2214BI2V — i2vLow (base model LOW pass)",
        "filename":    "smoothMixWan2214BI2V_i2vLow.safetensors",
        "dest_dir":    "diffusion_models",
        "search_url":  "https://civitai.com/search/models?query=smoothMixWan2214BI2V",
        "notes":       "Download the _i2vLow variant",
        "size_gb":     ~27,
    },
    {
        "description": "SVI Pro LoRA — HIGH (rank 128, fp16)",
        "filename":    "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
        "dest_dir":    "loras",
        "search_url":  "https://civitai.com/search/models?query=SVI_v2_PRO_Wan2.2",
        "notes":       "Download the HIGH variant",
        "size_gb":     ~1.5,
    },
    {
        "description": "SVI Pro LoRA — LOW (rank 128, fp16)",
        "filename":    "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors",
        "dest_dir":    "loras",
        "search_url":  "https://civitai.com/search/models?query=SVI_v2_PRO_Wan2.2",
        "notes":       "Download the LOW variant",
        "size_gb":     ~1.5,
    },
]


def print_banner():
    print()
    print("=" * 65)
    print("  Perry's Wan 2.2 Pipeline — Model Downloader")
    print("=" * 65)
    print()


def create_dirs():
    print("[SETUP] Creating model directories...")
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"        ✓  {path}")
    print()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def download_hf_models(hf_token: str = None):
    """Download all HuggingFace models."""
    print("=" * 65)
    print("  STEP 1 — Downloading HuggingFace Models")
    print("=" * 65)
    print()

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    for model in HF_MODELS:
        dest_dir   = DIRS[model["dest_dir"]]
        rename_to  = model.get("rename_to")
        filename   = model["filename"]
        final_name = rename_to if rename_to else Path(filename).name
        dest_path  = dest_dir / final_name

        print(f"  [{model['description']}]")
        print(f"  Repo   : {model['repo_id']}")
        print(f"  File   : {filename}")
        print(f"  Target : {dest_path}")

        if dest_path.exists():
            size = file_size_mb(dest_path)
            print(f"  Status : ✅ Already exists ({size:.1f} MB) — skipping")
            print()
            continue

        print(f"  Status : ⬇  Downloading...")
        try:
            downloaded = hf_hub_download(
                repo_id   = model["repo_id"],
                filename  = filename,
                local_dir = str(dest_dir),
                **kwargs,
            )
            # hf_hub_download may save with nested path — rename if needed
            downloaded_path = Path(downloaded)
            if downloaded_path.name != final_name:
                downloaded_path.rename(dest_path)

            size = file_size_mb(dest_path)
            print(f"  Status : ✅ Downloaded ({size:.1f} MB)")
        except Exception as e:
            print(f"  Status : ❌ FAILED — {e}")
            if model["required"]:
                print("           This model is REQUIRED. Pipeline will not work without it.")
        print()


def try_civitai_download(model: dict, token: str):
    """
    Attempt to auto-download a CivitAI model if a token and modelVersionId
    are available. Most users will need to download manually.
    Returns True on success, False if we should fall back to manual.
    """
    if not token:
        return False

    # CivitAI direct download requires the specific model version URL.
    # Without a hardcoded modelVersionId, we cannot auto-resolve it.
    # This is a placeholder for users who know their specific model version IDs.
    return False


def print_civitai_instructions():
    """Print manual download instructions for CivitAI models."""
    print()
    print("=" * 65)
    print("  STEP 2 — CivitAI Models (Manual Download Required)")
    print("=" * 65)
    print()
    print("  The following models must be downloaded manually from CivitAI.")
    print("  CivitAI requires a free account to download.")
    print()
    print("  HOW TO DOWNLOAD:")
    print("  1. Create a free account at https://civitai.com")
    print("  2. For each model below, visit the search URL")
    print("  3. Find the correct file, click Download")
    print("  4. Place each file in the directory shown")
    print()

    all_present = True
    for model in CIVITAI_MODELS:
        dest_dir   = DIRS[model["dest_dir"]]
        dest_path  = dest_dir / model["filename"]
        exists     = dest_path.exists()

        status = "✅ Already present" if exists else "❌ MISSING"
        if not exists:
            all_present = False

        size_note = f"(~{model['size_gb']} GB)" if not exists else f"({file_size_mb(dest_path):.1f} MB)"

        print(f"  ┌─ {model['description']}")
        print(f"  │  Status   : {status} {size_note}")
        print(f"  │  Filename : {model['filename']}")
        print(f"  │  Place in : {dest_dir}/")
        print(f"  │  Search   : {model['search_url']}")
        if model.get("notes"):
            print(f"  │  Note     : {model['notes']}")
        print(f"  └{'─' * 60}")
        print()

    if all_present:
        print("  ✅ All CivitAI models are already present!")
    else:
        print("  ⚠️  Download the missing models above before running the pipeline.")
    print()


def check_ask_gemini_node():
    """
    Check if the Ask_Gemini / MasterKey custom node is installed.
    This node's source repo is not publicly confirmed — instruct the user.
    """
    print("=" * 65)
    print("  STEP 3 — Ask_Gemini / MasterKey Node Check")
    print("=" * 65)
    print()

    comfyui_dir = BASE_DIR.parent / "ComfyUI"
    custom_nodes = comfyui_dir / "custom_nodes"

    if not custom_nodes.exists():
        print("  ⚠️  ComfyUI not found yet. Run setup_comfyui.sh first.")
        print()
        return

    # Search for the node class definitions
    found = False
    for py_file in custom_nodes.rglob("*.py"):
        try:
            text = py_file.read_text(errors="ignore")
            if "Ask_Gemini" in text or "MasterKey" in text:
                found = True
                print(f"  ✅ Found Ask_Gemini/MasterKey in: {py_file.parent.name}")
                break
        except Exception:
            continue

    if not found:
        print("  ❌ Ask_Gemini / MasterKey nodes NOT found in custom_nodes/")
        print()
        print("  HOW TO INSTALL:")
        print("  1. Open ComfyUI in browser at http://localhost:8188")
        print("  2. Click the 'Manager' button (top menu)")
        print("  3. Click 'Install Missing Nodes'")
        print("  4. Load any workflow JSON — Manager detects missing node types")
        print("     and shows the correct package to install")
        print()
        print("  OR: In your working installation, find the package by running:")
        print("  grep -r 'class Ask_Gemini\\|class MasterKey' ComfyUI/custom_nodes/")
        print("  Then: cd ComfyUI/custom_nodes/<that_folder> && git remote -v")
        print("  Add that URL to scripts/setup_comfyui.sh")
    print()


def verify_all():
    """Final summary of all required files."""
    print("=" * 65)
    print("  FINAL CHECK — All Required Model Files")
    print("=" * 65)
    print()

    all_required = [
        (DIRS["vae"]           / "wan_2.1_vae.safetensors",                                                    "VAE"),
        (DIRS["text_encoders"] / "umt5_xxl_fp16.safetensors",                                                  "Text Encoder"),
        (DIRS["diffusion_models"] / "smoothMixWan2214BI2V_i2vHigh.safetensors",                                "Base Model HIGH"),
        (DIRS["diffusion_models"] / "smoothMixWan2214BI2V_i2vLow.safetensors",                                 "Base Model LOW"),
        (DIRS["loras"]         / "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",             "LoRA HIGH"),
        (DIRS["loras"]         / "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors",              "LoRA LOW"),
    ]

    all_ok = True
    for path, label in all_required:
        if path.exists():
            size = file_size_mb(path)
            print(f"  ✅  {label:<25} {path.name}  ({size:.1f} MB)")
        else:
            print(f"  ❌  {label:<25} MISSING — {path}")
            all_ok = False

    print()
    if all_ok:
        print("  🎉 All models present. You are ready to run the pipeline!")
    else:
        print("  ⚠️  Some models are missing. See instructions above.")
    print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download models for Perry's Wan 2.2 Pipeline"
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var). "
             "Not required for public models but speeds up downloads.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which files exist; do not download anything.",
    )
    args = parser.parse_args()

    print_banner()
    create_dirs()

    if not args.check_only:
        download_hf_models(hf_token=args.hf_token)

    print_civitai_instructions()
    check_ask_gemini_node()
    ok = verify_all()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
