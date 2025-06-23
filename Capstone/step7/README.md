# Capstone Project Step 7: Cross-Modal Retrieval Experiments

> Automate, benchmark, and analyze cross-modal retrieval models that link real-world photos (MS-COCO, Flickr-30k) with Stable-Diffusion renders and their prompts—all from Jupyter notebooks.

---

## Overview
This project evaluates a spectrum of retrieval architectures, inversion models, and classical baselines.

* **Bridge real vs. AI** by comparing embeddings from COCO & Flickr-30k against Stable-Diffusion outputs.  
* **Support four retrieval modes**: *image ↔ image*, *text ↔ image*, *image → text*, *dataset ↔ dataset*.  
* **Assess prompt quality** via a closed-loop *image → prompt → image* cycle.  
* **Benchmark approaches**: TF-IDF + ResNet, OpenAI CLIP, OpenCLIP, Reverse-SD inversion, X-Modaler, HAT, DCLIP distillation.  
* **Automate experiments** with lightweight Hydra configs—even inside notebooks.  
* **Analyze metrics** (Recall@K, MRR, Median Rank, CLIPScore-Δ) plus model size, latency, and cost trade-offs.

---

## Project Structure

```bash
cross-modal-retrieval/
├── notebooks/
│   ├── 00_paths_build_metadata.ipynb     # data root + metadata.parquet
│   ├── 01_retrieval_dataset.ipynb        # PyTorch Dataset + sanity check
│   ├── 02_clip_baseline_training.ipynb   # zero-shot & finetune CLIP
│   ├── 03_metrics_utils.ipynb            # Recall@K, MRR, CLIPScore-Δ helpers
│   ├── 04_prompt_inversion.ipynb         # Reverse-SD experiments
│   ├── 05_advanced_models.ipynb          # X-Modaler, HAT, DCLIP, ensembles
│   └── 06_results_visualization.ipynb    # W&B dashboards, plots, tables
├── data/                  # metadata.parquet lives here
├── configs/               # Optional Hydra/YAML sweeps for batch runs
└── README.md              
