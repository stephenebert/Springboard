# Step 3: Problem Statement & Approach

## 1. Problem Statement

The goal is to build a **cross-modal retrieval system** that can compare and retrieve semantically similar entries between:

- Real-world images from MS COCO and Flickr-30k  
- AI-generated images and their prompts from Stable Diffusion  

This is inspired by the Kaggle competition:  
[Stable Diffusion – Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts)

We extend the competition by enabling **bi-directional retrieval** across both image and text modalities. Given any image or text, the system will retrieve semantically aligned counterparts from both **human-generated** and **AI-generated** datasets.

---

## 2. Importance

- **Bridges AI and Human Data**: Evaluates how closely synthetic data representations align with real-world images and human language.
- **Supports 4 Retrieval Modes**:
  - Image-to-Image
  - Text-to-Image
  - Image-to-Text
  - Dataset-to-Dataset
- **Prompt Fidelity Analysis**: Enables prompt reconstruction from generated images to assess generative model faithfulness.
- **Practical Applications**: Recommendation engines, image editing assistants, prompt engineering research, and benchmark datasets for future multimodal systems.

---

## 3. Data Overview

We’ll use three text-image datasets:

1. **Stable Diffusion (Prompt ↔ Image Pairs)**  
   - From the Kaggle competition above  
   - Synthetic prompts paired with AI-generated images

2. **MS COCO 2017 Captions**  
   - ~120K images with five human-written captions each  
   - Downloaded from: [MS COCO Dataset](https://cocodataset.org/#download)

3. **Flickr-30k Captions**  
   - ~31K images with natural language descriptions  
   - [Flickr-30k on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

---

## 4. ML Framing

| Question | Answer |
|----------|--------|
| **Supervised or Unsupervised?** | Retrieval is **unsupervised** (via cosine similarity); prompt inversion is **supervised** regression. |
| **Classification or Regression?** | Retrieval: no label prediction. Inversion: regression on CLIP text embeddings. |
| **Prediction Target** | Retrieval: ranked similarity. Inversion: vector prediction of prompt embedding. |
| **Features** | CLIP-based image/text embeddings. Possibly CNN features or prompt metadata for inversion. |
| **Approach** | Use pre-trained CLIP for feature extraction. Fine-tune a regression head for inversion. Compare to traditional ML baselines (TF-IDF, Ridge). |

---

## 5. Planned Capabilities

We’re building a **web-accessible search system** with:

- **Image-to-Image**: Find similar real or AI-generated images
- **Text-to-Image & Image-to-Text**: Search across modalities
- **Dataset-to-Dataset**: Explore semantic similarity between full datasets
- **Prompt Inversion**: Reconstruct likely prompts from AI-generated images

The system will be demoed through a **web UI** where users can upload images or input text and instantly retrieve results.

---

## 6. Infrastructure & Hardware Requirements

| Component | Specs |
|-----------|-------|
| **CPU** | 8-core / 16-thread modern processor for efficient I/O + API |
| **RAM** | ≥ 32 GB system RAM for embedding & caption batch ops |
| **GPU** | NVIDIA RTX-class GPU (≥ 16 GB VRAM) for CLIP encoding and Stable Diffusion inference |

---

