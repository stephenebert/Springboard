# Step 4: Survey, Reproduce, and Extend Existing Research

This step focuses on understanding prior work, collecting and evaluating public implementations, reproducing a baseline model, and planning extensions. It aligns with Step 4 of the Springboard capstone rubric.

---

## 4.1 Survey of Prior Research

We focus on CLIP-based retrieval, prompt inversion, and round-trip evaluation. Below are selected key papers:

| Title | Summary | Links |
|-------|---------|-------|
| **CLIP: Learning Transferable Visual Models From Natural Language Supervision**<br>Alec Radford *et al.* (2021) | Introduces CLIP, a contrastive pretraining method for vision-language tasks. Basis for zero-shot image-text retrieval, | [PDF](https://arxiv.org/abs/2103.00020), [GitHub](https://github.com/openai/CLIP) |
| **X-Modaler**<br>Yehao Li *et al.* (2021) | Unified framework for cross-modal analytics including retrieval and captioning. Built on CLIP. | [PDF](https://arxiv.org/abs/2108.08217), [GitHub](https://github.com/YehLi/xmodaler) |
| **HAT: Hierarchical Alignment Transformers**<br>Yi Bin *et al.* (2023) | Aligns image/text embeddings using shared Transformer backbones. | [PDF](https://arxiv.org/abs/2308.04343), [GitHub](https://github.com/LuminosityX/HAT) |
| **DCLIP: Distilled CLIP**<br>Daniel Csizmadia *et al.* (2025) | Improves CLIP via cross-modal distillation. | [PDF](https://arxiv.org/abs/2505.21549), [GitHub](https://anonymous.4open.science/r/DCLIP-B772/README.md) |
| **Reverse Stable Diffusion**<br>Florinel-Alin Croitoru *et al.* (2023) | Recovers prompts from generated images using regression + vocab classification. | [PDF](https://arxiv.org/abs/2308.01472), [GitHub](https://github.com/CroitoruAlin/Reverse-Stable-Diffusion) |

**Deliverable:** `papers/survey.md` (or `papers/papers.md`)

---

## 4.2 Public Code Implementations

For each key paper or related repo, we document:

- Name & citation
- Repo URL
- Setup & demo instructions
- Notes on quirks or compatibility

| Resource | URL | Quick Notes |
|----------|-----|-------------|
| **CLIP** (Radford et al., 2021) | [openai/CLIP](https://github.com/openai/CLIP) | `pip install -r requirements.txt` · `python demo.py` |
| **Reverse Stable Diffusion** (Croitoru et al., 2023) | [GitHub](https://github.com/CroitoruAlin/Reverse-Stable-Diffusion) | Requires Torch 2.0+ · Run with `invert.py` |
| **X-Modaler** | [GitHub](https://github.com/YehLi/xmodaler) | Modular CLIP tasks · Task-specific notebooks available |
| **HAT** | [GitHub](https://github.com/LuminosityX/HAT) | Transformer-only · Use `environment.yml` |
| **DCLIP** | [GitHub](https://anonymous.4open.science/r/DCLIP-B772/README.md) | Needs `diffusers >= 0.17` and `torch >= 2.1` |
| **Image-to-Prompts** | [GitHub](https://github.com/jacksonchen1998/Image-to-Prompts) | Inversion head demo + Colab |
| **Stable Diffusion Prompts** | [GitHub](https://github.com/MingyuanRen/Stable-Diffusion-Image-to-Prompts) | Uses VAE + language model decoder |

**Deliverable:** `implementations/implementations.md`

---

## 4.3 Reproduce a Baseline (CLIP Retrieval)

We will run a simple baseline retrieval system:

- **Task**: Image-to-text or text-to-image on COCO/Flickr captions using cosine similarity
- **Approach**: Encode all images and captions using CLIP; retrieve top-k by similarity

**Deliverable:**  
- `notebooks/reproduce_clip_baseline.ipynb`

---

## 4.4 Analyze & Reflect

Critically evaluate the reproduced results:

- Match to paper’s reported metrics (e.g., Recall@1, Recall@5)
- Identify performance gaps, edge cases, and limitations
- Note setup differences that might affect results

**Deliverable:**  
- `analysis/reproduction_analysis.md`

---

## 4.5 Synthesize Next Steps

Based on research and reproduction, we define a capstone-specific roadmap:

- Implement prompt inversion head with fine-tuned regression
- Add round-trip fidelity checks: (image → prompt → generated image → compare)
- Build search interface with all 4 retrieval modes

**Deliverables:**
- `roadmap/next_steps.md`
- Slide deck under `slides/`

---
