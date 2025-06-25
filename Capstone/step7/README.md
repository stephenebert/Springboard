# Cross-Modal Retrieval Capstone (Step 7)

A comprehensive pipeline for evaluating real vs. AI-generated images and text across four retrieval modes, with automated experiments, benchmarks, and visualization.

## Project Overview
Modern cross-modal models must bridge human–captured photos and generative outputs. This capstone:

* Aggregates three datasets: MS-COCO, Flickr-30k, and Stable Diffusion prompt–image pairs.
* Supports four retrieval modes:
    * **Image↔Image:** Nearest neighbors within and across domains.
    * **Text→Image:** Retrieve real photos and generate AI renders from text.
    * **Image→Text:** Produce captions or invert diffusion prompts.
    * **Dataset→Dataset:** Compare full caption corpora (COCO ↔ Flickr-30k).
* Benchmarks multiple architectures: ResNet+TF-IDF, OpenAI CLIP, OpenCLIP variants, X-Modaler, HAT, DCLIP.
* Automates reproducible runs via notebooks and a CLI (`run_experiment.py`).
* Evaluates: Recall@K, MRR, Median Rank, cross-domain recall, plus model size vs. embedding time trade-offs.
* Generalizes: cross-validation across seeds with error bars.

## Repository Structure
```
step7/
├── notebooks/                     # Jupyter workflows
│   ├── 00_build_metadata.ipynb    # Create metadata.parquet
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_clip_baseline.ipynb     # Single-model CLIP retrieval
│   ├── 03_multiple_backbones.ipynb# Run RN50, ViT-B-32, RN101
│   ├── 04_image2image.ipynb
│   ├── 05_text2image.ipynb
│   ├── 06_image2text.ipynb
│   ├── 07_dataset2dataset.ipynb
│   ├── 08_cost_performance.ipynb  # Model size vs. embed time
│   ├── 09_cross_validation.ipynb
│   └── 10_visual_demo.ipynb
├── src/                           # Python modules & utilities
│   ├── cross_modals.py            # Data loaders, transforms, metrics
│   └── retrieval.py               # Dataset class
├── run_experiment.py              # CLI wrapper for full pipeline
└── README.md                      # This file
```

## Setup & Installation
#### 1. Clone repository:
```bash
git clone <repo-url>
cd step7
```

#### 2. Install dependencies:
```bash
conda create -n xmodal python=3.10
conda activate xmodal
pip install -r requirements.txt
```

#### 3. Prepare data (edit `notebooks/00_build_metadata.ipynb` paths if needed):
* MS-COCO 2017 images & captions
* Flickr-30k images & captions
* Stable Diffusion prompt–image pairs

#### 4. Generate metadata:
```bash
jupyter nbconvert --to notebook --execute notebooks/00_build_metadata.ipynb
```

## Quickstart
#### Notebook Workflow
1.  **`02_clip_baseline.ipynb`**: Single-model retrieval with CLIP; set `BACKBONE` and run.
2.  **`03_multiple_backbones.ipynb`**: Batch-run RN50, ViT-B-32, RN101; embeds & metrics.
3.  **`04`-`07`**: Explore each retrieval mode (image↔image, text→image, image→text, dataset↔dataset).
4.  **`08_cost_performance.ipynb`**: Plot parameter count vs. embedding time.
5.  **`09_cross_validation.ipynb`**: Cross-validation error bars.
6.  **`10_visual_demo.ipynb`**: Qualitative nearest-neighbor grids.

#### CLI Alternative
```bash
python run_experiment.py \
  --model ViT-B-32 \
  --pretrained openai \
  --preset cpu-fast \
  --max-samples 10000 \
  --batch 64
```
This command will create a new folder under `experiments/` containing:
* `img_embs.npy`, `txt_embs.npy`
* `metrics.json` (Recall@1/5/10, median rank)
* `config.json` (hyperparams, param_count, embed_secs)

## Key Results

| Model    | Params (M) | Embed Time (min) | R@1   | R@5   | R@10  |
| :------- | :--------: | :--------------: | :---- | :---- | :---- |
| RN50     |     50     |       3.0        | 43.3% | 66.6% | 75.3% |
| ViT-B-32 |    151     |       8.0        | 67.0% | 90.2% | 95.9% |
| RN101    |    100     |       5.0        | 33.3% | 56.3% | 65.8% |

-   **Cross-domain recall@1 (image→image):** ~24%
-   **Prompt inversion R@1:** ~45%
-   **Cross-val std < 1.5%** across 5 seeds

