## Step 4 Plan: Survey, Reproduce, and Extend Existing Research

We’ll tackle Step 4 in five focused sub-steps. Each section below describes **what** to do, **how**, and **where** to record the work.

---

### 4.1 Survey Existing Research  
**Objective:** Build a solid understanding of prior work on CLIP-based retrieval, prompt inversion, and round-trip retrieval.  
**Actions:**  
- Select key papers; for each write a 1-2 paragraph summary covering problem, approach, results, and links.  
- **Core papers to include**:  
  - **Learning Transferable Visual Models From Natural Language Supervision**  
    Alec Radford *et al.* (2021)  
    Introduces CLIP, a contrastive language–image pretraining method that set the foundation for zero-shot cross-modal retrieval.  
    [PDF](https://arxiv.org/abs/2103.00020) | [GitHub](https://github.com/openai/CLIP)  
  - **X-modaler: A Versatile and High-performance Codebase for Cross-modal Analytics**  
    Yehao Li *et al.* (2021)  
    Provides a unified framework (built on CLIP and related models) for tasks including image↔text retrieval, captioning, VQA, and more.  
    [PDF](https://arxiv.org/abs/2108.08217) | [GitHub](https://github.com/YehLi/xmodaler)  
  - **Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval (HAT)**  
    Yi Bin *et al.* (2023)  
    Proposes Hierarchical Alignment Transformers (HAT), using identical Transformer backbones for both image and text to improve retrieval alignment.  
    [PDF](https://arxiv.org/abs/2308.04343) | [GitHub](https://github.com/LuminosityX/HAT)  
  - **Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation**  
    Daniel Csizmadia *et al.* (2025)  
    A teacher–student distillation framework that refines CLIP embeddings through cross-modal transformer distillation to boost retrieval metrics.  
    [PDF](https://arxiv.org/abs/2505.21549)  
    Code & checkpoints: <https://anonymous.4open.science/r/DCLIP-B772/README.md>  
  - **Reverse Stable Diffusion: What prompt was used to generate this image?**  
    Florinel-Alin Croitoru *et al.* (2023)  
    The first method for inverting a diffusion model to recover its original text prompt via joint regression and vocabulary classification.  
    [PDF](https://arxiv.org/abs/2308.01472) | [GitHub](https://github.com/CroitoruAlin/Reverse-Stable-Diffusion)  
**Deliverable:**  
- `papers/survey.md` (or append to `papers/papers.md`)

---

### 4.2 Document Public Implementations  
**Objective:** Gather and vet available codebases for the selected papers and extra Image→Prompt resources.  
**Actions:**  
- For each core paper (and extra repos), record:  
  1. **Name & reference**  
  2. **Repo URL**  
  3. **Install steps** (e.g. `pip install -r requirements.txt`)  
  4. **How to run a minimal demo**  
  5. **Any quirks or notes**  
**Deliverable:**  
- `implementations/implementations.md` with entries like:

| Resource                                                         | Repo URL                                                                 | Notes / Quick Start                                                                                            |
|------------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **CLIP** (Radford et al., 2021)                                  | https://github.com/openai/CLIP                                           | `pip install -r requirements.txt`; run `python demo.py --image … --text …`                                      |
| **Reverse Stable Diffusion** (Croitoru et al., 2023)             | https://github.com/CroitoruAlin/Reverse-Stable-Diffusion                 | Includes pretrained weights; requires Torch 2.0+; `python invert.py --image_path sd.png`                       |
| **X-modaler** (Ye Li et al., 2021)                                | https://github.com/YehLi/xmodaler                                         | Unified CLIP wrapper; `pip install -r requirements.txt`; see `README.md` for task-specific notebooks          |
| **HAT** (Yi Bin et al., 2023)                                     | https://github.com/LuminosityX/HAT                                        | Transformer‐only backbone; `conda env create -f environment.yml`; `python run_retrieval.py --config …`         |
| **DCLIP** (Csizmadia et al., 2025)                                | https://anonymous.4open.science/r/DCLIP-B772/README.md                    | Teacher–student distillation; follow link for code + checkpoints; may need diffusers 0.17+ and PyTorch 2.1      |
| **Image-to-Prompts** (Jackson Chen)                              | https://github.com/jacksonchen1998/Image-to-Prompts?tab=readme-ov-file    | PyTorch-based inversion head; `pip install -r requirements.txt`; demo Colab link in repo overview              |
| **Stable-Diffusion-Image-to-Prompts** (Mingyuan Ren)              | https://github.com/MingyuanRen/Stable-Diffusion-Image-to-Prompts?utm…     | Uses VAE features + LM decoding; `pip install -r requirements.txt`; run `python infer_prompt.py --img …`       |
---

### 4.3 Reproduce One Baseline End-to-End  
**Objective:** Verify I can run a core retrieval pipeline for myself.  
**Actions:**  
- Choose a straightforward baseline (e.g. CLIP nearest-neighbor on COCO captions).  
- Adapt paths, run it, capture metric (e.g. Recall@1) and example retrievals.  
**Deliverable:**  
- `notebooks/reproduce_clip_baseline.ipynb`

---

### 4.4 Analyze & Reflect  
**Objective:** Critically assess how well the baseline matches published results and what I learned.  
**Actions:**  
- Compare my reproduced metric to the paper’s reported value.  
- Note discrepancies, root causes, strengths, and limitations.  
**Deliverable:**  
- `analysis/reproduction_analysis.md`

---

### 4.5 Synthesize Next Steps  
**Objective:** Define how the capstone will extend these foundations.  
**Actions:**  
- Outline enhancements (e.g. prompt-inversion head, round-trip eval).  
- Draft a roadmap for Step 5+.  
**Deliverable:**  
- `roadmap/next_steps.md`  
- slide deck in `slides/`

---
