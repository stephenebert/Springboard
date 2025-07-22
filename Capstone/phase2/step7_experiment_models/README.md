# Step 7: Experiment With Various Models

A comprehensive pipeline for evaluating real vs. AI-generated images and text across four retrieval modes, with automated experiments, benchmarks, and visualization.

---

## Project Overview

This capstone project:

- Aggregates three datasets: **MS-COCO**, **Flickr-30k**, and **Stable Diffusion prompt–image pairs**
- Supports four cross-modal retrieval modes:
  - **Image ↔ Image**
  - **Text → Image**
  - **Image → Text**
  - **Dataset → Dataset**
- Benchmarks multiple architectures (e.g. ResNet+TF-IDF, OpenAI CLIP, OpenCLIP, X-Modaler, HAT, DCLIP)
- Automates reproducible runs via Jupyter notebooks and CLI (`run_experiment.py`)
- Evaluates metrics: **Recall@K**, **MRR**, **Median Rank**, and **cost-performance tradeoffs**
- Includes cross-validation and error analysis

---

## Repository Structure

```
step7/
├── notebooks/
│   ├── 00_build_metadata.ipynb
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_clip_baseline.ipynb
│   ├── 03_multiple_backbones.ipynb
│   ├── 04_image2image.ipynb
│   ├── 05_text2image.ipynb
│   ├── 06_image2text.ipynb
│   ├── 07_dataset2dataset.ipynb
│   ├── 08_cost_performance.ipynb
│   ├── 09_cross_validation.ipynb
│   └── 10_visual_demo.ipynb
├── src/
│   ├── cross_modals.py
│   └── retrieval.py
├── run_experiment.py
├── experiments/
├── Figure_1.png
├── step_7_notes.pdf
└── README.md
```

---

## Setup & Installation

```bash
git clone <repo-url>
cd step7

conda create -n xmodal python=3.10
conda activate xmodal
pip install -r requirements.txt
```

Update data paths in `notebooks/00_build_metadata.ipynb`, then run:

```bash
jupyter nbconvert --to notebook --execute notebooks/00_build_metadata.ipynb
```

---

## Quickstart

### Notebook Workflow
1. `02_clip_baseline.ipynb`: Single-model CLIP retrieval
2. `03_multiple_backbones.ipynb`: Batch-run RN50, ViT-B-32, RN101
3. `04–07`: Each retrieval mode
4. `08_cost_performance.ipynb`: Model size vs. embedding time
5. `09_cross_validation.ipynb`: Cross-validation
6. `10_visual_demo.ipynb`: Nearest-neighbor grids

### CLI Alternative

```bash
python run_experiment.py \
  --model ViT-B-32 \
  --pretrained openai \
  --preset cpu-fast \
  --max-samples 10000 \
  --batch 64
```

Creates outputs in `experiments/`:
- `img_embs.npy`, `txt_embs.npy`
- `metrics.json`, `config.json`

---

## Key Results

| Model    | Params (M) | Embed Time (min) | R@1   | R@5   | R@10  |
|----------|------------|------------------|-------|-------|-------|
| RN50     | 50         | 3.0              | 43.3% | 66.6% | 75.3% |
| ViT-B-32 | 151        | 8.0              | 67.0% | 90.2% | 95.9% |
| RN101    | 100        | 5.0              | 33.3% | 56.3% | 65.8% |

- **Cross-domain image→image R@1**: ~24%
- **Prompt inversion R@1**: ~45%
- **Cross-validation std < 1.5%**

---

## 📈 Retrieval Mode Results

### 1. Image ↔ Image (Cross-Domain)
- R@1: ~24%
- R@5: ~55%
- R@10: ~67%

### 2. Text → Image
- R@1: ~48%
- R@5: ~77%
- R@10: ~85%

### 3. Image → Text
- R@1: ~53%
- R@5: ~78%
- R@10: ~88%

### 4. Dataset → Dataset
- COCO→Flickr R@1: ~0%
- Flickr→COCO R@1: ~0–0.2%

---

## Cross-Validation (ViT-B-32, 5 seeds)

| Seed | R@1   | R@5   | R@10  |
|------|-------|-------|--------|
| 0    | 43.10 | 66.53 | 75.31  |
| 1    | 42.29 | 66.25 | 74.88  |
| 2    | 43.04 | 66.62 | 74.84  |
| 3    | 42.34 | 65.47 | 74.11  |
| 4    | 42.63 | 66.80 | 75.90  |

**Summary:**

| Metric | Mean  | Std   |
|--------|-------|-------|
| R@1    | 42.68 | 0.38  |
| R@5    | 66.33 | 0.52  |
| R@10   | 75.01 | 0.66  |

> Performance is stable across seeds with <1% variance, indicating strong reliability.

---

## Example Output

![Figure 1: Qualitative Nearest-Neighbor Retrieval](Figure_1.png)  
*Nearest-neighbor retrieval grids across real and AI-generated domains.*

---

## Documentation

- [`step_7_notes.pdf`](step_7_notes.pdf): Full explanation of architecture choices, metric definitions, and experimental setup.

---
