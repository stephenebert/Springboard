# Capstone Project Step 7: Cross-Modal Retrieval Experiments

> Automate, benchmark, and analyze multiple cross-modal retrieval models on image-caption datasets using Jupyter notebooks.

## Overview
This project systematically evaluates a suite of retrieval architectures, inversion models, and traditional baselines. Our goals are to:

- **Bridge real vs. AI domains** by comparing embeddings from COCO and Flickr-30k against Stable Diffusion outputs.  
- **Support four retrieval modes**: image↔image, text↔image, image→text, and dataset↔dataset.  
- **Assess prompt quality** via a closed-loop image→prompt→image pipeline.  
- **Benchmark diverse approaches**: CLIP, TF-IDF baselines, Reverse Diffusion inversion, X-Modaler, HAT, DCLIP distillation, plus any new open-source models.  
- **Automate** reproducible pipelines for training, evaluation, and logging.  
- **Analyze** quantitative metrics (Recall@K, MRR, Median Rank), plus model-size, latency, and cost trade-offs.

## Project Structure
```bash
project/
├── notebooks/             # Jupyter workflows
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_embedding_extraction.ipynb
│   ├── 03_retrieval_baselines.ipynb
│   ├── 04_prompt_inversion.ipynb
│   ├── 05_advanced_models.ipynb
│   └── 06_evaluation_visualization.ipynb
├── src/                   # Helper modules (loaders, models, metrics)
│   ├── data/              # Parsing & splitting logic
│   ├── features/          # Embedding extraction scripts
│   └── models/            # Model definitions
├── configs/               # Hydra/YAML experiment configs
├── experiments/           # Logs & checkpoints (optional)
└── README.md              # This file
