# Step 2: Data Collection & Exploratory Data Analysis (EDA)

This directory documents the **data collection** and **exploratory analysis** efforts for the capstone project: *Reverse Prompt Engineering for Stable Diffusion Images*. This step is aligned with Springboard’s Capstone Phase 1, Step 2.

---

## Objective

To collect high-quality datasets relevant to image-to-text modeling tasks, unify their schema, and analyze linguistic and structural patterns through exploratory data analysis (EDA).

---

## Directory Structure

- `data/`: Raw and processed datasets (captions, metadata).
- `data-collection/`: Scripts or instructions to download data (Kaggle CLI, COCO tools, etc.).
- `explore_stable_diffusion.ipynb`: Main EDA notebook.
- `README.md`: Current file with overview of Step 2 goals and findings.

---

## Datasets Used

1. **Stable Diffusion - Image to Prompts**  
   - [Kaggle competition page](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/data)  
   - Thousands of image-prompt pairs (AI-generated)  
   - Used as core dataset for reverse engineering

2. **MS COCO 2017 Captions**  
   - [COCO Captions](https://cocodataset.org/#download)  
   - Human-written captions for 123K real-world images  
   - Used for linguistic baselines and Zipf-style analysis

3. **Flickr-30k Captions**  
   - [Flickr-30k on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data)  
   - 30K images with 5 captions each  
   - Complements COCO for style and diversity comparison

---

## Key EDA Topics

The notebook `explore_stable_diffusion.ipynb` contains:

1. **Schema Harmonization**
   - Normalizes caption fields (`image_id`, `caption`, etc.)
2. **Quality Checks**
   - Missing value detection, duplicates, outlier lengths
3. **Linguistic Stats**
   - Word counts, sentence lengths, punctuation/digit frequency
4. **Visualizations**
   - Histograms, word clouds, token distributions
5. **Feature Engineering**
   - Sentence complexity, vocabulary richness, repetition
6. **Zipf’s Law Verification**
   - Plots word frequency vs. rank, checks log-log linearity

---

## Note on Data Hosting

Large files (>100MB) are excluded from GitHub. Please:
- Use Kaggle CLI or provided URLs to download datasets
- Consider Git LFS or an external link (e.g., Google Drive or AWS S3) for reproducibility

---

## Summary

The collected datasets are well-suited for training reverse prompt models. Both COCO and Flickr-30k exhibit strong natural language properties and follow Zipf's Law. This validates the data quality and ensures compatibility with NLP-based modeling.

