# Capstone Project Step 7: Cross-Modal Retrieval Experiments

> Automate, benchmark, and analyze multiple cross-modal retrieval models on image-caption datasets.

## Overview
Step 7 builds on data wrangling from Step 5 and baseline reproduction Step 4 to systematically evaluate a suite of retrieval architectures:

- **Benchmark**: Compare CLIP, Reverse Stable Diffusion inversion, X-Modaler, HAT, DCLIP, and any additional open-source models.
- **Automate**: Scripted pipelines for training, evaluation, and logging—ensuring reproducibility.
- **Analyze**: Quantitative metrics (Recall@K, MRR, Median Rank), plus model-size and latency trade-offs.

## Repository Structure
```
step7_experiments/
├── configs/               # YAML configs for each experiment
│   ├── clip.yaml
│   ├── xmodaler.yaml
│   └── hat.yaml
├── scripts/               # Core Python scripts
│   ├── run_experiment.py  # Train + eval one config
│   ├── collect_results.py # Gather metrics into a CSV
│   └── plot_results.py    # Generate summary figures
├── notebooks/             # Exploratory notebooks
│   └── 07_experiments.ipynb
└── README.md              
```

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```  
Requirements include PyTorch (≥1.10), Transformers, pandas, matplotlib, and tqdm.

### 2. Configure an Experiment
Each YAML in `configs/` defines:
- `model` & `backbone` (e.g., `openai/clip-vit-base-patch32`)
- Data paths (`train` & `val` parquet files)
- Training hyperparameters (`batch_size`, `lr`, `epochs`)
- Similarity (`pooling`, `metric`) and ranking `metrics` (e.g., `recall@1`, `mrr`)

```yaml
# example: configs/clip.yaml
model: clip
backbone: openai/clip-vit-base-patch32

data:
  train: /data/coco_train.parquet
  val:   /data/coco_val.parquet

training:
  batch_size: 64
  lr: 1e-4
  epochs: 5

similarity:
  pooling: mean
  metric: cosine

metrics:
  - recall@1
  - recall@5
  - mrr
```

### 3. Run an Experiment
```bash
python scripts/run_experiment.py \
  --config configs/clip.yaml \
  --output_dir runs/clip_baseline
```
This will train for the specified epochs, evaluate on validation data each epoch, and save per-epoch metrics to `runs/clip_baseline/metrics_epoch{n}.json`.

### 4. Aggregate & Visualize Results
```bash
# 4.1 Collect all metrics
python scripts/collect_results.py --runs_dir runs --out metrics_summary.csv

# 4.2 Plot comparison charts
python scripts/plot_results.py \
  --metrics metrics_summary.csv \
  --out figures/
```
Generated figures (e.g., `recall_comparison.png`, `mrr_comparison.png`) will appear under `figures/`.

## Evaluation Metrics
- **Recall@K**: Portion of queries whose true match ranks in the top K (e.g., Recall@1, Recall@5).
- **Mean Reciprocal Rank (MRR)**: Average of 1 / (rank of correct item).
- **Median Rank**: Median position of the correct match—robust to outliers.

## Adding New Models
1. Add a new YAML under `configs/` (e.g., `dclip.yaml`).
2. Ensure any extra dependencies are in `requirements.txt`.
3. Run with:
   ```bash
   python scripts/run_experiment.py --config configs/dclip.yaml --output_dir runs/dclip
   ```

## Reproducibility & Tips
- All runs are tied to the Git commit; include config filename in logs.
- For larger hyperparameter sweeps, integrate **Optuna** or **Ray Tune**, and log to W&B or MLflow.
