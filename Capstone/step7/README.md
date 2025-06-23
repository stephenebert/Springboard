# Capstone Project Step 7: Cross-Modal Retrieval Experiments

> Automate, benchmark, and analyze multiple cross-modal retrieval models on image-caption datasets linking real-world photos (MS-COCO, Flickr-30k) with Stable-Diffusion renders and their prompts.

## Overview
This repository systematically evaluates a suite of retrieval architectures, inversion models, and traditional baselines. Our objectives are to

- **Bridge real vs. AI domains** by comparing embeddings from COCO and Flickr-30k against Stable-Diffusion outputs.  
- **Support four retrieval modes**: *image↔image*, *text↔image*, *image→text*, and *dataset↔dataset*.  
- **Assess prompt quality** through a closed-loop *image→prompt→image* cycle.  
- **Benchmark diverse approaches**: TF-IDF + ResNet, OpenAI CLIP, OpenCLIP, Reverse-SD inversion, X-Modaler, Hierarchical Alignment Transformer (HAT), and DCLIP distillation.  
- **Automate** reproducible pipelines for training, evaluation, and logging with Hydra / PyTorch Lightning.  
- **Analyze** quantitative metrics—Recall@K, MRR, Median Rank, CLIPScore-Δ—alongside model-size, latency, and cost trade-offs.

## Project Structure
```bash
cross-modal-retrieval/
├── notebooks/                     # Jupyter workflows
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_embedding_extraction.ipynb
│   ├── 03_retrieval_baselines.ipynb
│   ├── 04_prompt_inversion.ipynb
│   ├── 05_advanced_models.ipynb
│   └── 06_evaluation_visualization.ipynb
├── src/                           # Helper modules (loaders, models, metrics)
│   ├── data/                      # Parsing & splitting logic
│   ├── datasets/                  # PyTorch Dataset / DataModule classes
│   ├── models/                    # CLIP, X-Modaler, HAT, DCLIP, Reverse-SD
│   ├── trainers/                  # Lightning Trainer & callbacks
│   ├── metrics.py                 # Retrieval & loop metrics
│   ├── utils.py
│   └── run.py                     # Hydra entry-point
├── configs/                       # YAML / Hydra experiment configs
├── experiments/                   # Auto-logged runs & checkpoints
├── data/                          # Symlinked or local datasets
├── docs/                          # Diagrams, analysis slides, report draft
└── README.md                      
