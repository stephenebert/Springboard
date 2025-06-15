# Step 5: Data Wrangling

In this step we collected, cleaned, and merged three datasets for cross-modal retrieval. All code and narrative are implemented in the Jupyter notebook:

- **Raw sources**:  
  1. **Stable Diffusion Prompts**  
     - Source: [Kaggle – Stable Diffusion Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/data)  
     - File: `custom_prompts_df.csv`  
  2. **MS COCO 2017 Captions**  
     - Source: [MS COCO Captions Dataset](https://cocodataset.org/#download)  
     - Files: `captions_train2017.json` / `captions_val2017.json`  
  3. **Flickr-30k Captions**  
     - Source: [Kaggle – Flickr-30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data)  
     - File: `results.csv`

- **Cleaning**:  
  1. Dropped any rows missing `image_id` or `caption`.  
  2. Enforced max-token lengths (SD ≤100, COCO ≤50, F30k ≤50), dropping over-length entries.  
  3. Lowercased, stripped whitespace, and deduplicated exact `(image_id, caption)` pairs.  
  4. Saved cleaned tables as `sd_clean.parquet`, `coco_clean.parquet`, `f30k_clean.parquet` on Desktop.

- **Merged dataset**:  
  - Concatenated the three cleaned tables, added a `source` column.  
  - Ensured `image_id` is string-typed.  
  - Saved final `crossmodal_dataset.parquet` to the Desktop.
