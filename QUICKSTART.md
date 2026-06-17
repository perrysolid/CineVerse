

> Assumes ComfyUI already installed. 24GB+ VRAM recommended.
> 📄 Full details in `README.md` inside the zip.

---

## 1. Download Models

### HuggingFace — paste into browser or `huggingface-cli`

```
VAE → https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
      └── wan_2.1_vae.safetensors (~400 MB)
      └── Place in: ComfyUI/models/vae/

Text Encoder → https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
               └── umt5_xxl_fp16.safetensors (~10 GB)
               └── Place in: ComfyUI/models/text_encoders/
```

### CivitAI — free account required

```
Diffusion models → ComfyUI/models/diffusion_models/

  https://civitai.com/models/1995784/smooth-mix-wan-22-14b-i2vt2v?modelVersionId=2260110
  └── smoothMixWan2214BI2V_i2vHigh.safetensors (~27 GB)

  https://civitai.com/models/1995784/smooth-mix-wan-22-14b-i2vt2v?modelVersionId=2259006
  └── smoothMixWan2214BI2V_i2vLow.safetensors (~27 GB)

LoRAs → ComfyUI/models/loras/

  https://civitai.com/search/models?query=SVI+v2+PRO+Wan+2.2+I2V
  └── SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors 
  └── SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors 
```

### Low VRAM alternative (optional, quality reduction applies)

```
GGUF quantized → https://huggingface.co/Bedovyy/smoothMixWan22-I2V-GGUF/tree/main
                 └── Place in: ComfyUI/models/diffusion_models/
```

> We used **NVIDIA A100-SXM4-80GB** for all submitted generations.
> Primary BF16 models are preferred — alternatives give slightly lower quality.

---

## 2. Custom Nodes

Open **ComfyUI Manager → Install Missing Nodes**, drag any workflow `.json` onto the canvas — Manager auto-detects all required nodes.

Manual install if needed:

```
ComfyUI Manager      → https://github.com/Comfy-Org/ComfyUI-Manager
KJNodes              → search in Manager
VideoHelperSuite     → search in Manager  (ComfyUI-VideoHelperSuite)
ComfyRoll            → search in Manager
Ask_Gemini/MasterKey → auto-detected when workflow is loaded
```

---

## 3. Prompt-Enhancement Stage

```
The MasterKey node holds a category-specialist system prompt.
It reads your reference image + short instruction and rewrites
them into a structured paragraph tuned for the UMT5-XXL encoder.

Configure the prompt-enhancement node once in ComfyUI as described
by its node package — no changes are needed per run.
```

Keep your instruction short; the MasterKey does the structural expansion.

---

## 4. Pick Your Workflow

| Image content | Workflow file |
|---------------|---------------|
| Jewelry, rings, watches | `jewellery_workflow.json` |
| Person with face or torso | `people_workflow.json` |
| Phone, laptop, tablet | `tech_workflow.json` |
| Hands only / object physics | `close_workflow.json` |
| Unsure / mixed | `universal_workflow.json` |

Always use the specialist workflow for known content — Universal is a fallback only.

---

## 5. Resolution & Frames — Speed vs Quality

Default output: **720p · 161 frames · ~10 seconds**

### Switch to 480p — in the `ImageResizeKJv2` node

```
Landscape workflows (People / Tech / Close / Universal):
  Width  1280 → 854
  Height  720 → 480

Jewellery portrait workflow:
  Width   404 → 228
  Height  720 → 405
```
--ASPECT RATIO REMAINS SAME.

### Reduce frame count — in the scheduler / sampler node

```
161 frames = ~10 s  (default, competition standard)
 97 frames = ~6 s   saves ~40% inference time
 65 frames = ~4 s   saves ~60% inference time
```

> Identity fidelity metrics are per-frame — reducing frame count does not affect per-frame scores.

### VRAM & time reference

```
A100 80GB  → 720p · 161f · BF16            (workflow dependent)
RTX 3090   → 480p · 161f · BF16          
RTX 3080   → 480p ·  97f · FP8 models    
16 GB GPU  → 480p ·  65f · FP8 + lowvram 
```

Add `--lowvram` in `scripts/start_comfyui.sh` for GPUs under 24 GB.
See **Section 10** and **Section 15** of `README.md` for FP8 sources and full VRAM table.

---

## 6. Run

```
1. Drag workflow .json onto ComfyUI canvas
2. LoadImage node    → drop your reference image
3. Prompt node       → type a short motion instruction (see examples below)
                    → set mode to fixed, note the seed
4. Queue Prompt  (Ctrl+Enter)
```

**Instruction examples:**

```
Jewellery : "Camera slowly orbits the gold ring, highlighting the diamond facets."
People    : "Person slices carrots on a wooden cutting board with deliberate strokes."
Tech      : "Hand picks up the iPhone 15 Pro, tilts to show the titanium back."
Close     : "Scissors open and cut through bubble wrap, popping cells as they close."
Universal : "The ring rotates slowly on a velvet stand under studio lighting."
```

Keep instructions short — the MasterKey stage expands them into the full structured prompt.
Output saves to `ComfyUI/output/` — h264-mp4, CRF 19.

---

## 7. Reproducibility

To reproduce any generation exactly:

```bash
python3 run_inference.py \
    --workflow workflows/jewellery_workflow.json \
    --image    inputs/my_image.jpg \
    --prompt   "Camera slowly orbits the gold ring." \
    --seed     42
```

Every run logs seed + job ID automatically to `seed_log.txt`.
Omit `--seed` to generate a random seed — it is printed and saved.

---

## 8. How It Works

```
Your short instruction
        ↓
  MasterKey prompt-enhancement stage
  (category-specialist system prompt — rewrites your instruction
   into a structured paragraph optimised for UMT5-XXL attention)
        ↓
  Wan 2.2  TWO-PASS diffusion
  ├── HIGH model  → locks structure, identity, motion trajectory
  └── LOW model   → refines texture, sharpness, temporal smoothness
        ↓
  SVI Pro LoRA on both passes  (temporal stability, reduces flicker)
        ↓
  Output video  720p · 161 frames · 16 fps · h264-mp4
```

> **RegexReplace nodes** are present on the canvas but bypassed — not active in current pipeline.
> Can be activated for multi-scene video extension, but requires restructuring MasterKey prompts into scene-wise demarcations. See `README.md` for details.

---

> 📄 Full workflow settings, MasterKey internals, token weighting, evaluation criteria mapping, FP8 setup, and troubleshooting → **`README.md`**
