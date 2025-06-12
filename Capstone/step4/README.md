## Step 4 Plan: Survey, Reproduce, and Extend Existing Research

We’ll tackle Step 4 in five focused sub-steps. Each section below describes **what** to do, **how**, and **where** to record the work.

---

### 4.1 Survey Existing Research  
**Objective:** Build a solid understanding of prior work on CLIP-based retrieval, prompt inversion, and round-trip retrieval.  
**Actions:**  
- Select 3–5 key papers; for each write a 1–2 paragraph summary covering problem, approach, results, and links.  
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
**Objective:** Gather and vet available codebases for the selected papers.  
**Actions:**  
- Find GitHub repos, Colabs, or Docker images.  
- Note for each: clone URL, install steps, run instructions, and any setup quirks.  
**Deliverable:**  
- `implementations/implementations.md`

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
- 3–5 slide deck in `slides/`

---
