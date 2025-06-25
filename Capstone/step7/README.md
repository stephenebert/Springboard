# Experiment Pipeline

This repository contains the code and notebooks to run a series of retrieval experiments, from data preparation to evaluation.

## Running Experiments

### 1. Build Metadata
Execute `00_paths_build_metadata.ipynb` to parse COCO, Flickr, and Stable Diffusion datasets into a single `metadata.parquet` file. This is the first step before running any retrieval model.

### 2. Baseline Retrieval
This step trains and evaluates the baseline CLIP models.

-   **Single Backbone:**
    -   Open `02_clip_baseline_training.ipynb`.
    -   Set the `BACKBONE` variable (e.g., `"RN50"`, `"ViT-B-32"`).
    -   Run the notebook end-to-end to generate embeddings and results.

-   **Multiple Backbones:**
    -   For a streamlined process, run the script `02b_multiple_backbones.py`.
    -   This will generate runs for `RN50`, `ViT-B-32`, and `RN101` with a single command.

### 3. Specialized Modes
Run these notebooks to test specific retrieval scenarios.

-   **Image → Image:** `04_image_to_image_retrieval.ipynb`
-   **Text → Image:** `05_text_to_image_retrieval.ipynb`
-   **Image → Text:** `05b_image_to_text_retrieval.ipynb`
-   **Dataset → Dataset:** `07_dataset_to_dataset.ipynb`

### 4. Evaluation & Visualization
Analyze the results from the experiments.

-   **Scaling & Cost Analysis:** `06_evaluation_visualization.ipynb`
-   **Cross-validation:** `11_cross_validation.ipynb`
-   **Ablation Studies:** `08_model_ablation.ipynb`
-   **Qualitative Grids:** `09_visual_nn_demo.ipynb` (for visual nearest neighbor demos)

### 5. Command-Line Interface (CLI) Alternative
Instead of using the notebooks, you can run the full pipeline via the command line. This will produce an `experiments/` subfolder containing embeddings, metrics, and configuration files.

```bash
python run_experiment.py --model ViT-B-32 --preset cpu-fast --max 10000
```

## Results & Findings

-   **Recall@1/5/10:** `ViT-B-32` achieves **~67%/90%/96%** on 10k samples; `RN50` achieves **~43%/66%/75%**; `RN101` achieves **~33%/56%/66%**.

-   **Generalization:** Cross-validation standard deviation is less than **1.5%** across different seeds on 5k subsamples, indicating stable performance.

-   **Cost Tradeoff:**
    -   `ViT-B-32` (~150M params): ~8 minutes to embed 10k samples.
    -   `RN50` (~50M params): ~3 minutes to embed 10k samples.
    -   `RN101` (~100M params): ~5 minutes to embed 10k samples.

-   **Prompt Inversion:** Recovering Stable Diffusion prompts from images yields a **~45%** recall@1.
