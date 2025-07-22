# Step 5: Data Wrangling

In this step, we **collected, cleaned, and unified three datasets** of image–caption pairs for downstream cross-modal retrieval experiments.

---

## Directory Structure

```
step5_data_wrangling/
├── data/                     # Contains raw, cleaned, and merged datasets (Parquet format)
└── notebooks/
    └── 01_data_wrangling.ipynb  # Core data wrangling notebook
```

---

## Raw Data Sources

We gathered three distinct datasets to ensure diversity across real and synthetic image–text pairs:

1. **Stable Diffusion Prompts**  
   - Synthetic prompt–image pairs from Kaggle's competition  
   - File: `custom_prompts_df.csv`  
   - [Kaggle link](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/data)

2. **MS COCO 2017 Captions**  
   - Human-written captions for real-world images  
   - Files: `captions_train2017.json`, `captions_val2017.json`  
   - [COCO Captions](https://cocodataset.org/#download)

3. **Flickr-30k Captions**  
   - 31K images with 5 captions each  
   - File: `results.csv`  
   - [Flickr-30k on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data)

---

## 🔍 Data Inspection

Each dataset was examined for:
- Missing fields (`image_id`, `caption`)
- Caption-length distributions
- Zipf’s Law behavior in word frequency (log-log plots)

---

## Cleaning Pipeline

A standardized cleaning process was applied across all datasets:

1. **Missing Values**  
   - Dropped entries missing either `image_id` or `caption`

2. **Length Filtering**  
   - Kept only prompts with ≤ 100 tokens (Stable Diffusion)  
   - Kept captions with ≤ 50 tokens (COCO, Flickr-30k)

3. **Normalization**  
   - Lowercased all text  
   - Trimmed whitespace from captions/prompts

4. **Deduplication**  
   - Removed duplicate `(image_id, caption)` pairs

---

## Cleaned Output

Each cleaned dataset was saved as a `.parquet` file for efficient downstream processing:

| Dataset          | File Name            | Cleaned Rows |
|------------------|----------------------|--------------|
| Stable Diffusion | `sd_clean.parquet`   | 100,000      |
| MS COCO          | `coco_clean.parquet` | 591,555      |
| Flickr-30k       | `f30k_clean.parquet` | 158,784      |

---

## Merged Dataset

The cleaned datasets were:
- Combined with a new `source` column (`SD`, `COCO`, or `F30k`)
- Cast `image_id` to string type for compatibility
- Concatenated into a unified dataframe with **850,339 rows**

Final output file: `crossmodal_dataset.parquet`

---

## Summary

This step produced a high-quality, large-scale cross-modal dataset spanning real and synthetic sources. The merged dataset supports flexible retrieval experiments across domains and modalities, and is compatible with vector search and CLIP-based pipelines.

---

