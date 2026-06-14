#!/usr/bin/env python3
"""
Perry's Wan 2.2 Pipeline — Inference Script
============================================
Runs a ComfyUI workflow from the command line with a fixed, recorded seed.
Deterministic: same --seed + same --image + same --prompt = identical output.

USAGE
-----
  python3 run_inference.py \\
      --workflow  workflows/jewellery_workflow.json \\
      --image     inputs/my_ring.jpg \\
      --prompt    "Camera slowly orbits the gold ring, highlighting the diamond facets." \\
      --seed      42 \\
      --output    outputs/

  # Omit --seed to generate a random seed (value printed and saved to seed_log.txt)
  python3 run_inference.py \\
      --workflow  workflows/universal_workflow.json \\
      --image     inputs/product.jpg \\
      --prompt    "Object rotates slowly on a clean surface."

REQUIREMENTS
------------
  ComfyUI must already be running:
    bash scripts/start_comfyui.sh

  All model files must be present (verify with):
    python3 scripts/verify_setup.py

REPRODUCIBILITY
---------------
  Every run appends a line to seed_log.txt:
    <timestamp>  workflow=<name>  image=<file>  seed=<value>  job_id=<id>

  To exactly reproduce a previous run, pass its seed with --seed <value>.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
import uuid
from datetime import datetime
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("COMFYUI_PORT", 8188))
SEED_LOG = "seed_log.txt"

# Node title patterns used to locate the Ask_Gemini node inside a workflow JSON.
# The node that receives both the image and the motion instruction.
ASK_GEMINI_TITLES = {"Ask_Gemini", "AskGemini", "ask_gemini"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def comfy_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def check_server(host: str, port: int) -> bool:
    """Return True if ComfyUI is reachable."""
    try:
        urllib.request.urlopen(comfy_url(host, port, "/system_stats"), timeout=5)
        return True
    except Exception:
        return False


def upload_image(host: str, port: int, image_path: str) -> str:
    """Upload a local image to ComfyUI and return the server-side filename."""
    import mimetypes
    import io

    image_path = Path(image_path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        image_data = f.read()

    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        comfy_url(host, port, "/upload/image"),
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["name"]


def patch_workflow(workflow: dict, image_name: str, prompt: str, seed: int) -> dict:
    """
    Patch the workflow JSON in-place:
      - Set LoadImage filename to the uploaded image.
      - Set the motion instruction text in the Ask_Gemini node.
      - Set the seed in the Ask_Gemini node and switch mode to 'fixed'.

    Returns the patched workflow dict.
    """
    patched = json.loads(json.dumps(workflow))  # deep copy

    for node_id, node in patched.items():
        class_type = node.get("class_type", "")
        title = node.get("_meta", {}).get("title", "")

        # ── LoadImage node ────────────────────────────────────────
        if class_type == "LoadImage":
            node["inputs"]["image"] = image_name
            print(f"  [patch] Node {node_id} (LoadImage) → image = {image_name}")

        # ── Ask_Gemini node ───────────────────────────────────────
        if class_type in ASK_GEMINI_TITLES or title in ASK_GEMINI_TITLES:
            inputs = node.setdefault("inputs", {})

            # Motion instruction goes into the first text widget (field index varies
            # by node version; try common key names in order of preference).
            for key in ("text", "prompt", "instruction", "query"):
                if key in inputs:
                    inputs[key] = prompt
                    print(f"  [patch] Node {node_id} (Ask_Gemini) → {key} = {prompt[:60]}...")
                    break
            else:
                # Fallback: set whichever string field comes first
                for key, val in inputs.items():
                    if isinstance(val, str) and len(val) > 5:
                        inputs[key] = prompt
                        print(f"  [patch] Node {node_id} (Ask_Gemini) → {key} (fallback) set")
                        break

            # Seed + deterministic mode
            inputs["seed"] = seed
            inputs["mode"] = "fixed"
            print(f"  [patch] Node {node_id} (Ask_Gemini) → seed = {seed}, mode = fixed")

    return patched


def queue_prompt(host: str, port: int, workflow: dict) -> str:
    """Submit the workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({"prompt": workflow, "client_id": uuid.uuid4().hex}).encode()
    req = urllib.request.Request(
        comfy_url(host, port, "/prompt"),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def wait_for_completion(host: str, port: int, prompt_id: str, poll_interval: int = 5):
    """Poll /history until the job is complete. Print progress dots."""
    print(f"\n  Queued job {prompt_id}")
    print("  Waiting for completion", end="", flush=True)
    while True:
        time.sleep(poll_interval)
        try:
            url = comfy_url(host, port, f"/history/{prompt_id}")
            with urllib.request.urlopen(url, timeout=10) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed", False):
                    print(" ✓")
                    return history[prompt_id]
                if status.get("status_str") == "error":
                    print(" ✗")
                    raise RuntimeError(f"ComfyUI reported an error for job {prompt_id}. "
                                       "Check the ComfyUI terminal for details.")
        except urllib.error.URLError:
            pass  # server briefly busy — keep polling
        print(".", end="", flush=True)


def log_seed(workflow_name: str, image_path: str, seed: int, job_id: str):
    """Append a reproducibility record to seed_log.txt."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (f"{timestamp}  workflow={workflow_name}  "
            f"image={Path(image_path).name}  seed={seed}  job_id={job_id}\n")
    with open(SEED_LOG, "a") as f:
        f.write(line)
    print(f"\n  Seed logged → {SEED_LOG}")
    print(f"  {line.strip()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a Perry Wan 2.2 workflow with a fixed, recorded seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workflow", required=True,
                        help="Path to workflow JSON (e.g. workflows/jewellery_workflow.json)")
    parser.add_argument("--image",    required=True,
                        help="Path to reference image (any resolution — pipeline resizes)")
    parser.add_argument("--prompt",   required=True,
                        help="Short motion instruction, e.g. 'Camera orbits the ring slowly.'")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Fixed integer seed for deterministic output. "
                             "Omit to generate a random seed (printed + logged).")
    parser.add_argument("--host",     default=DEFAULT_HOST, help=f"ComfyUI host (default: {DEFAULT_HOST})")
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT,
                        help=f"ComfyUI port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────
    workflow_path = Path(args.workflow)
    image_path    = Path(args.image)

    if not workflow_path.exists():
        sys.exit(f"[ERROR] Workflow not found: {workflow_path}")
    if not image_path.exists():
        sys.exit(f"[ERROR] Image not found: {image_path}")

    # ── Seed ──────────────────────────────────────────────────────
    if args.seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"\n  No --seed supplied. Generated random seed: {seed}")
        print(f"  Pass --seed {seed} to reproduce this exact run.")
    else:
        seed = args.seed
        print(f"\n  Using fixed seed: {seed}")

    # ── Server check ──────────────────────────────────────────────
    print(f"\n  Connecting to ComfyUI at {args.host}:{args.port} ...")
    if not check_server(args.host, args.port):
        sys.exit(
            f"[ERROR] ComfyUI is not reachable at {args.host}:{args.port}.\n"
            "        Start it first:  bash scripts/start_comfyui.sh"
        )
    print("  ✓ ComfyUI is online")

    # ── Load workflow ─────────────────────────────────────────────
    with open(workflow_path) as f:
        workflow = json.load(f)
    print(f"\n  Workflow : {workflow_path.name}")
    print(f"  Image    : {image_path}")
    print(f"  Prompt   : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  Seed     : {seed}")

    # ── Upload image ──────────────────────────────────────────────
    print("\n  Uploading reference image ...")
    server_image_name = upload_image(args.host, args.port, str(image_path))
    print(f"  ✓ Uploaded as: {server_image_name}")

    # ── Patch workflow ────────────────────────────────────────────
    print("\n  Patching workflow nodes ...")
    patched = patch_workflow(workflow, server_image_name, args.prompt, seed)

    # ── Queue ─────────────────────────────────────────────────────
    job_id = queue_prompt(args.host, args.port, patched)

    # ── Wait ──────────────────────────────────────────────────────
    wait_for_completion(args.host, args.port, job_id)

    # ── Log seed ──────────────────────────────────────────────────
    log_seed(workflow_path.name, str(image_path), seed, job_id)

    print("\n  ✓ Generation complete.")
    print(f"  Output video saved to ComfyUI output directory.")
    print(f"  To reproduce: python3 run_inference.py "
          f"--workflow {workflow_path} --image {image_path} "
          f"--prompt \"{args.prompt}\" --seed {seed}\n")


if __name__ == "__main__":
    main()
