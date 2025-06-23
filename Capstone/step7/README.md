# Capstone Project — Step 7: Cross-Modal Retrieval Experiments

> Automate, benchmark, and analyze multiple cross-modal retrieval models on image-caption datasets.

## Overview
This step builds on data wrangling and baseline reproduction to systematically evaluate a suite of retrieval architectures, inversion models, and traditional baselines. Our goals are to:

- **Bridge real vs. AI domains** by comparing embeddings from COCO and Flickr-30k against Stable Diffusion outputs.  
- **Support four retrieval modes**: image↔image, text↔image, image→text, and dataset↔dataset.  
- **Assess prompt quality** via a closed-loop image→prompt→image pipeline.  
- **Benchmark diverse approaches**: CLIP, TF-IDF baselines, Reverse Diffusion inversion, X-Modaler, HAT, DCLIP distillation, plus any new open-source models.  
- **Automate** reproducible pipelines for training, evaluation, and logging.  
- **Analyze** quantitative metrics (Recall@K, MRR, Median Rank), plus model-size, latency, and cost trade-offs.

## Project Structure
```bash
project/
├── data/                  # Raw & processed datasets
│   ├── coco/              # COCO images & captions
│   ├── flickr30k/         # Flickr-30k images & captions
│   └── sd_prompts/        # Stable Diffusion prompt-image pairs
├── src/                   # Source code
│   ├── data/              # Download, preprocessing, splits
│   ├── features/          # Embedding extraction scripts
│   ├── models/            # Baseline & DL model definitions
│   ├── train.py           # Train & validation entry point
│   └── evaluate.py        # Retrieval & inversion evaluation scripts
├── configs/               # YAML configs for each experiment (Hydra)
├── experiments/           # Logs, checkpoints, metrics outputs
├── notebooks/             # EDA & result visualization notebooks
└── README.md              # Project overview and instructions
