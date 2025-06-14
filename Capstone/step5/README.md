# Step 5: Data Wrangling

In this step, we collect and clean the datasets needed for cross-modal retrieval and prompt inversion:

- **data/raw/**: source files (COCO captions JSON, Flickr30k CSV, SD prompts CSV/JSON)
- **data/cleaned/**: cleaned, filtered, and merged dataset for downstream experiments
- **notebooks/01_data_wrangling.ipynb**: notebook with all wrangling code and narrative

We will inspect schemas, handle missing values & outliers, apply cleaning rules, and produce a unified dataset.
