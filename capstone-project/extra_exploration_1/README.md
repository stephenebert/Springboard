# ✨ Stable Diffusion v1.5 — Text → Image Demo

[![HF Space](https://img.shields.io/badge/🤗 Space-click%20to%20try-blue?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/<YOUR-HF-HANDLE>/sd15-t2i-demo)

Turn any prompt into a 512 × 512 image using **Stable Diffusion v1.5** (🤗 **diffusers**) wrapped in a clean **Gradio** UI.  
Runs on CPU, CUDA, **or Apple Silicon (M-series Metal)**.

![UI](images/bear%20walking%20in%20SD.png)

---
## Overview
1. One-file Gradio app (```text2image_demo.py```)
   - Detects your compute backend (CUDA, Apple Metal / MPS, or CPU).
   - Pulls Stable Diffusion v1.5 from the Hugging Face Hub the first time you run it (cached afterwards).
3. Minimal UI
   Delivered with Gradio:

| Control                    | Purpose                                                                          |
| -------------------------- | -------------------------------------------------------------------------------- |
| **Prompt** textbox         | The text you want to turn into an image.                                         |
| **Inference Steps** slider | How many denoising steps to run (≈ quality vs. speed).                           |
| **Guidance Scale** slider  | “CFG” scale—how strongly the model follows your prompt.                          |
| **Seed** field             | `0` or blank → random; any other int means *re-generate exactly the same image*. |

A two-column Gallery on the right shows each image as soon as it’s finished and lets you download it.

![UI](images/cyber%20punk%20SD.png)


3. Under the hood: the SD pipeline
```
Text Prompt ──▶ CLIP Text Encoder ──▶ Text Embedding
                                            │
                                            ▼
                                      Scheduler (DDIM) ──▶ Iterative Denoising
                                            │                in Latent Space
                                            ▼                      │
                                      Random Noise ──▶ UNet ◀─────┘
                                            │         (guided by text embedding)
                                            ▼
                                      Final Latent ──▶ VAE Decoder ──▶ 512×512 RGB Image
```
- Inference Steps = number of DDIM iterations.
- Guidance Scale mixes the unconditional and conditional UNet predictions (higher = stricter prompt following).
- The seed is fed to the scheduler’s RNG; same seed + prompt + params = identical image.

4. Zero bulky repo assets
No checkpoints inside the repo—only ~50 lines of Python plus a few screenshots.
Requirements are light (< 300 MB once the SD weights are cached).

5. Cross-platform

- CUDA: full FP16 SD inference.
- Apple Silicon: Metal acceleration automatically triggered (```torch.backends.mps```)
- CPU: still works; just slower.

6. Directory
```
extra_exploration_1/
├─ images/                  # screenshots for the README
├─ text2image_demo.py       # the Gradio app
├─ requirements.txt         # 6 core deps (torch, diffusers …)
├─ pyaudioop.py             # shim so Gradio ≥4.28 installs cleanly on Python 3.13
└─ README.md                # quick-start, examples, feature table
```
End-to-end flow

1. ```pip install -r requirements.txt```
2. ```python text2image_demo.py```
3. Browser opens → type prompt → click Generate → watch the latent noise resolve into an image (progress bar & live updates).

You can run this locally or package into a Hugging Face Space.

---

## Features

| Ready  | What                                                  |
|:--------:|-------------------------------------------------------|
| ✔️       | Prompt textbox + sliders (steps, CFG scale)            |
| ✔️       | Optional deterministic seed                           |
| ✔️       | Two-column **Gallery** with download buttons          |
| ✔️       | Auto-detects GPU (CUDA or MPS)                        |
| ✔️       | Zero bulky assets – model is pulled & cached automatically |

---

## Quick Start

```bash
git clone https://github.com/<your-handle>/stable-diffusion-t2i-demo.git
cd stable-diffusion-t2i-demo

python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt

python text2image_demo.py          # http://127.0.0.1:7860
```
![UI](images/terminal.png)
---

## requirements.txt
```
torch>=2.2
diffusers>=0.28
transformers>=4.42
accelerate>=0.29
safetensors
gradio>=4.28
pyaudioop ; python_version >= "3.13"   # optional shim for Gradio’s audio import
```

## Performance Table

| Device           | Steps | Time   |
| ---------------- | ----- | ------ |
| **M2 Max (mps)** | 50    | \~9 s  |
| RTX 3080 10 GB   | 50    | \~4 s  |
| 8-core CPU       | 50    | \~50 s |

Measured with Torch 2.2 + Diffusers 0.28

## Acknowledgements
- Stable Diffusion v1.5 — CompVis, Runway, Stability AI, LAION
- diffusers, transformers, gradio — Hugging Face
