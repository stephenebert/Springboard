# Springboard's MI/AI Bootcamp

Welcome to the Springboard MI/AI Bootcamp portfolio repository! This repo contains all my completed mini-projects and the final capstone project from the Springboard Machine Learning / Artificial Intelligence Bootcamp. Each project demonstrates key ML/AI concepts through hands-on exploration using Python, Pandas, Scikit-Learn, TensorFlow/Keras, Neural Networks, Deep Learning, Computer Vision, Natural Language Processing, Generative AI, Model Deployment, Diffusers, AWS, Gradio, and more.


## Contents

**Capstone**

This repository’s capstone is a two‑part, cross‑modal retrieval and generation system that showcases modern AI pipelines end‑to‑end:

-**Image-to-Text Retrieval Service**

-- Uses BLIP to generate captions for any uploaded image.

-- Encodes captions with CLIP into 512‑dim embeddings.

-- Performs ultra‑fast nearest‑neighbor search over a pre‑built FAISS index of MS‑COCO captions.

-- Exposes a FastAPI backend (```/health```, ```/search```) and a Gradio front‑end (Gallery UI) deployed on Hugging Face Spaces.

- **Text-to-Image Model‑Switcher Demo**

-- Wraps multiple Stable Diffusion checkpoints (v1.5, SDXL Base 1.0, SD‑Turbo) in a single Gradio app.

-- Auto‑detects compute backend (CUDA, Apple MPS, or CPU).

-- Uses ```DPMSolverMultistepScheduler``` for faster sampling and supports reproducible seeds.

-- Live‑switch models at runtime and view results immediately.

**Key Capabilities**:

1. End‑to‑end pipeline from raw input (image or text) through model inference and user‑friendly UI.

2. Production‑grade deployment with caching, token‑based access, and Hugging Face Space integration.

3. Performance benchmarks demonstrating speed vs. quality trade‑offs across models and hardware.
   
> See the full capstone code, assets, Docker configurations, and benchmarking scripts under the capstone/ directory.


### Mini-Projects
A collection of focused notebooks, each exploring a core ML technique:
| Project Title | Description |
|---------------|-------------|
| `Mini_Project_Build_a_Machine_Learning_Model.ipynb` | Build and evaluate a supervised ML model pipeline. |
| `Mini_Project_Building_a_Recommendation_Engine.ipynb` | Collaborative and content-based filtering on MovieLens data. |
| `Mini_Project_Building_a_Flask_Application_for_a_Machine_Learning_Model.ipynb` | Wrap a trained ML model in a Flask API for inference. |
| `Mini_Project_End_to_end_Churn_Prediction_Using_SageMaker.ipynb` | Full churn-prediction pipeline deployed with AWS SageMaker. |
| `Mini_Project_Exploratory_Data_Analysis.ipynb` | EDA: data cleaning, visualization, & summary statistics. |
| `Mini_Project_Fine_tuning_a_Convolutional_Neural_Network.ipynb` | Transfer learning with VGG/ResNet on an image classification task. |
| `Mini_Project_Logistic_Regression.ipynb` | Implement and interpret logistic regression for classification. |
| `Mini_Project_ML_Model.ipynb` | End-to-end ML pipeline: preprocessing, modeling, and evaluation. |
| `Mini_Project_More_Experience_With_Machine_Learning.ipynb` | Deeper dives into advanced ML techniques and pipelines. |
| `Mini_Project_Trees_and_Forests.ipynb` | Train and compare decision trees and random forests. |

## Getting Started

### Prerequisites
- Python 3.8+ (recommended 3.9 or 3.10)
- Anaconda or ```venv``` for environment management
- Git

### Setup 
1. Clone the repository:
``` bash
git clone https://github.com/<your-username>/Springboard-MI-AI-Bootcamp.git
cd Springboard-MI-AI-Bootcamp
```
2. Create and activate your environment
``` bash
# using conda
conda create -n sb-mlai python=3.9
conda activate sb-mlai

# or using venv
python3 -m venv venv
source venv/bin/activate
```
3. Run Jupyter notebooks:
   ``` bash
   jupyter lab  # or jupyter notebook
   ``` 
   
