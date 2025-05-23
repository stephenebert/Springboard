# Step 2: Data Collection & Exploratory Data Analysis (EDA)

This repository documents the **data collection** and **exploratory analysis** process for the Capstone Project, as part of the ML & AI Engineering Bootcamp.

---

## Objective

The goal of this stage is to collect and evaluate datasets relevant to prompt-to-image modeling tasks, and to explore their structure, quality, and linguistic characteristics through rigorous EDA. This step aligns with the Capstone Step 2 rubric criteria including completeness, process understanding, and presentation quality.

---

## Contents

- `data/`: Directory housing the downloaded raw datasets.
- `data-collection/`: Contains scripts or references for downloading or organizing datasets.
- `explore_stable_diffusion.ipynb`: The main Jupyter notebook for loading, inspecting, and analyzing the datasets.
- `README.md`: Describes the objective, methodology, and findings.

---

## Datasets Used

Three datasets were collected and explored:

1. **Stable Diffusion Prompts**  
   - Source: [Kaggle - Stable Diffusion Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/data)  

2. **MS COCO 2017 Captions**  
   - Source: [MS COCO Captions Dataset](https://cocodataset.org/#download)  

3. **Flickr-30k Captions**  
   - Source: [Kaggle - Flickr-30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data)  
   
---

## Key EDA Topics Covered

The `explore_stable_diffusion.ipynb` notebook contains:

1. **Dataset Loading and Schema Unification**  
   Ensures consistent fields like `image_id` and `caption`.

2. **Quality Checks**  
   Missing value analysis, duplicate removal, and column validation.

3. **Descriptive Statistics**  
   Includes caption length analysis, character count, and vocabulary size.

4. **Distributions & Visualizations**  
   Histograms of token lengths and word clouds per dataset.

5. **Feature Engineering & Correlation**  
   Examines relationships between length, punctuation, digits, etc.

6. **Zipf’s Law Analysis**  
   Investigates the power-law distribution of word frequencies with slope estimation and visual confirmation.

---

## Data Source Links

- [Stable Diffusion Dataset on Kaggle](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/data)
- [MS COCO 2017 Captions](https://cocodataset.org/#download)
- [Flickr-30k Dataset on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data)

---

## Notes on Hosting Large Files

For datasets over 100MB, local `.csv` and image files are excluded from GitHub tracking. Please use the original links to download the data or configure Git LFS / S3 as needed. The smaller caption CSVs (MS COCO & Flickr-30k) are already committed under the `data/` directory.

---

## Summary

This step confirms that the collected datasets are suitable for further modeling, meet scale requirements, and exhibit robust language characteristics. We confirm that both COCO and Flickr-30k follow Zipf’s Law, validating their linguistic fidelity.

