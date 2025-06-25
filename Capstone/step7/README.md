# Cross-Modal Retrieval Capstone (Step 7)

This project provides a unified pipeline to automate, benchmark, and analyze multiple cross-modal retrieval models on image–caption datasets, covering real photos (MS-COCO, Flickr-30k) and Stable Diffusion renders & prompts.

## Overview

This project builds a unified pipeline for evaluating cross-modal retrieval across four modes:

* **Image→Image:** Given an image (real or AI-generated), retrieve its nearest neighbors from both domains.
* **Text→Image:** Given a text prompt, fetch real-world images and generate new ones via Stable Diffusion.
* **Image→Text:** Generate descriptive captions or reconstruct original prompts from images.
* **Dataset→Dataset:** Compare entire corpora (e.g., Flickr-30k captions ↔ MS-COCO captions).

#### Key Components:
* **Retrieval baselines:** TF-IDF + ResNet, OpenAI CLIP, OpenCLIP.
* **Prompt inversion:** Reverse Stable Diffusion to recover original prompts.
* **Advanced models:** X-Modaler, HAT, DCLIP distillation.
* **Automated experiments:** Hydra/Lightning & CLI (`run_experiment.py`).
* **Metrics:** Recall@K, MRR, Median Rank, cross-domain recall, plus model size & embedding time.
* **Generalization:** Cross-validation across random seeds with error-bar analysis.

## Repository Structure
```
step7/
├── 00_paths_build_metadata.ipynb      # Build metadata from COCO/Flickr/Stable Diffusion
├── 01_retrieval_dataset.ipynb         # Dataset classes & sampling
├── 01a_image_text_paired.ipynb        # Exploratory data inspection
├── 02_clip_baseline_training.ipynb    # Single-backbone CLIP embedding + Recall@K
├── 02b_multiple_backbones.py          # Loop over RN50, ViT-B-32, RN101
├── 03_metrics_utils.ipynb             # Utility functions for metrics & loops
├── 04_image_to_image_retrieval.ipynb  # Nearest-neighbor image search demos
├── 05_text_to_image_retrieval.ipynb   # Text→image (real + AI) retrieval demos
├── 05b_image_to_text_retrieval.ipynb  # Image→text (captioning + prompt inversion)
├── 06_evaluation_visualization.ipynb  # Plot scaling, ablation, cost vs. performance
├── 07_dataset_to_dataset.ipynb        # Corpus-level retrieval analysis
├── 08_model_ablation.ipynb            # Ablation studies across architectures
├── 09_visual_nn_demo.ipynb            # Qualitative nearest-neighbor grids
├── 10_Run_Experiment.ipynb            # CLI-based reproducible experiment
├── 11_cross_validation.ipynb          # Cross-val variance with precomputed embeddings
├── cross_modals.py                    # Shared helper functions
├── run_experiment.py                  # CLI entry point for full pipeline
└── Figure_1.png                       # Sample qualitative result
```

## Setup & Dependencies

#### 1. Clone & Unzip
```bash
git clone <your-repo-url>
unzip step7.zip -d cross-modal-retrieval
cd cross-modal-retrieval/step7
```

#### 2. Create Environment & Install
```bash
conda create -n xmodal python=3.10
conda activate xmodal
pip install -r requirements.txt
```

#### 3. Prepare Data
Ensure your data is structured under the following paths:
```
C:/Users/steph/OneDrive/Desktop/data/metadata.parquet
C:/Users/steph/OneDrive/Desktop/data/SD/images
C:/Users/steph/OneDrive/Desktop/data/coco/train2017
C:/Users/steph/OneDrive/Desktop/data/flickr30k/flickr30k_images
```

## 🚀 Running Experiments

#### 1. Build Metadata
Run `00_paths_build_metadata.ipynb` to generate `metadata.parquet`.

#### 2. Baseline Retrieval
-   **Single backbone:** Open `02_clip_baseline_training.ipynb`, set the `BACKBONE`, and run.
-   **Multiple backbones:**
    ```bash
    python 02b_multiple_backbones.py
    ```

#### 3. Specialized Modes
-   **Image→Image:** `04_image_to_image_retrieval.ipynb`
-   **Text→Image:** `05_text_to_image_retrieval.ipynb`
-   **Image→Text:** `05b_image_to_text_retrieval.ipynb`
-   **Dataset→Dataset:** `07_dataset_to_dataset.ipynb`

#### 4. Evaluation & Visualization
-   **Scaling & cost:** `06_evaluation_visualization.ipynb`
-   **Cross-validation:** `11_cross_validation.ipynb`
-   **Ablation:** `08_model_ablation.ipynb`
-   **Qualitative grids:** `09_visual_nn_demo.ipynb`

#### 5. CLI Alternative
```bash
python run_experiment.py --model ViT-B-32 --preset cpu-fast --max 10000
```

## 📊 Results & Findings

-   **Recall@K:**
    -   `ViT-B-32`: **~67%/90%/96%** on 10k samples
    -   `RN50`: **~43%/66%/75%**
    -   `RN101`: **~33%/56%/66%**

-   **Generalization:** Cross-validation standard deviation is less than **1.5%** across 5 seeds.

-   **Cost Tradeoff:**
    -   `ViT-B-32` (~150M parameters) → ~8 min to embed 10k samples
    -   `RN50` (~50M) → ~3 min
    -   `RN101` (~100M) → ~5 min

-   **Prompt Inversion:** **~45% R@1** when recovering Stable Diffusion prompts.

## 📞 License & Contact
-   **Author:** Your Name (`your.email@example.com`)
-   **License:** This project is licensed under the **MIT License**.
