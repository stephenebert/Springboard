# Step 5: Data Wrangling

In this step we collected, cleaned, and merged three datasets for cross-modal retrieval:

- **Raw sources** (in Step 2):  
  - COCO captions JSON (`captions_train2017.json` / `captions_val2017.json`)  
  - Flickr30k captions (`results.csv`)  
  - Stable Diffusion prompts CSV (`custom_prompts_df.csv`)  

- **Cleaning**:  
  1. Dropped any rows missing `image_id` or `caption`.  
  2. Enforced max-token lengths (SD ≤100, COCO ≤50, F30k ≤50), dropping over-length entries.  
  3. Lowercased, stripped whitespace, and deduplicated exact `(image_id, caption)` pairs.  
  4. Saved cleaned tables as `sd_clean.parquet`, `coco_clean.parquet`, `f30k_clean.parquet` on Desktop.

- **Merged dataset**:  
  - Concatenated the three cleaned tables, added a `source` column.  
  - Ensured `image_id` is string-typed.  
  - Saved final `crossmodal_dataset.parquet` to your Desktop.
