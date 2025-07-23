# Springboard's MI/AI Bootcamp

Welcome to the Springboard MI/AI Bootcamp portfolio repository! This repo contains all my completed mini-projects and the final capstone project from the Springboard Machine Learning / Artificial Intelligence Bootcamp. Each project demonstrates key ML/AI concepts through hands-on exploration using Python, Pandas, Scikit-Learn, TensorFlow/Keras, Neural Networks, Deep Learning, Computer Vision, Natural Language Processing, Generative AI, Model Deployment, Diffusers, AWS, Gradio, and more.


# AI Vision and Generation Capstone

A comprehensive two-part, cross-modal retrieval and generation system demonstrating modern AI pipelines from end-to-end. This capstone project showcases AI capabilities through two interconnected systems that bridge the gap between visual understanding and text generation. Built with production-grade architecture and deployed on modern cloud infrastructure.

## System Architecture

### Part 1: [Image-to-Text Retrieval Service](https://huggingface.co/spaces/stephenebert/image2text-faiss-demo)

![Screenshot of the Gradio demo UI](Capstone/extra_exploration/data/UI1.png)


**Transform images into meaningful text through intelligent search**

- **Visual Understanding**: Leverages BLIP (Bootstrapped Language-Image Pre-training) to generate rich, contextual captions from any uploaded image
- **Semantic Encoding**: Utilizes CLIP embeddings to convert captions into high-dimensional 512-vector representations
- **Lightning-Fast Search**: Implements FAISS (Facebook AI Similarity Search) indexing over MS-COCO dataset for ultra-fast nearest-neighbor retrieval
- **Production API**: 
  - FastAPI backend with health monitoring (`/health`) and search endpoints (`/search`)
  - Interactive Gradio frontend with gallery UI
  - Deployed and accessible via Hugging Face Spaces

### Part 2: [Text-to-Image Model Switcher Demo](https://huggingface.co/spaces/stephenebert/model-switcher-sd)

![SD UI](Capstone/extra_exploration_1/bear%20walking%20in%20SD.png)


**Generate visuals from text descriptions with multiple AI models**

- **Multi-Model Support**: Seamlessly switch between industry-leading Stable Diffusion variants:
  - Stable Diffusion v1.5 (balanced quality and speed)
  - SDXL Base 1.0 (enhanced detail and resolution)
  - SD-Turbo (optimized for rapid generation)
- **Smart Hardware Detection**: Automatically detects and optimizes for available compute:
  - CUDA (NVIDIA GPUs)
  - Apple Metal Performance Shaders (Apple Silicon)
  - CPU fallback support
- **Advanced Sampling**: Implements `DPMSolverMultistepScheduler` for superior image quality with faster generation times
- **Reproducible Results**: Full seed control for consistent, repeatable outputs
- **Real-Time Switching**: Dynamic model switching without restart requirements

## Key Features

### Production-Ready Architecture
- End-to-end pipeline processing from raw inputs to polished user interfaces
- Robust caching mechanisms for improved performance
- Token-based access control and security
- Seamless Hugging Face Spaces integration

### Performance Optimization
- Comprehensive benchmarking suite comparing speed vs. quality trade-offs
- Hardware-specific optimizations across different compute backends
- Efficient memory management and model loading strategies

### User Experience
- Intuitive Gradio interfaces for both services
- Real-time results and immediate feedback
- Gallery view for image browsing and comparison
- Live model performance metrics

## Performance Benchmarks

### FAISS Index Performance on MS-COCO Dataset (591,753 embeddings)

Comprehensive analysis of different FAISS indexing strategies for production-scale similarity search:

| Index Type | Build Time | Size (MB) | Query Latency | Throughput (QPS) | Recall@10 |
|------------|------------|-----------|---------------|------------------|-----------|
| **FlatL2** | 0.07s | 1,155.77 | 0.31ms | 3,205 | 99.0% |
| **IVF_1024** | 0.49s | 1,162.29 | 0.01ms | **160,486** | 99.0% |
| **IVF_4096** | 1.71s | 1,168.31 | 0.00ms | **290,183** | 99.0% |

**Key Findings:**
- **50x Speed Improvement**: IVF_1024 delivers 50× faster queries than exact search with zero recall loss
- **Production Sweet Spot**: IVF_1024 offers optimal balance of build time, memory usage, and query performance
- **High-Concurrency Ready**: IVF_4096 achieves 290K QPS for applications requiring maximum throughput
- **Tail Latency Analysis**: p99 latency stays under 20ms across all configurations

### Stable Diffusion Generation Performance

Cross-platform performance analysis for text-to-image generation:

| Hardware Platform | Model | Resolution | Generation Time | Memory Usage |
|-------------------|-------|------------|-----------------|--------------|
| **NVIDIA RTX 3080** | SD v1.5 | 512×512 | 4-8s | ~4GB VRAM |
| **Apple M2 Max** | SD v1.5 | 512×512 | 12-20s | ~6GB RAM |
| **CPU (16-core)** | SD v1.5 | 512×512 | 60s+ | ~8GB RAM |
| **NVIDIA RTX 3080** | SDXL Base | 1024×1024 | 8-15s | ~8GB VRAM |

**Optimization Highlights:**
- **FP16 Acceleration**: 2x speedup on compatible hardware
- **Dynamic Scheduling**: DPMSolverMultistepScheduler reduces steps by 30-50%
- **Memory Efficiency**: Smart model loading and unloading for resource-constrained environments

## Technical Stack

- **Deep Learning**: PyTorch, Transformers, Diffusers
- **Computer Vision**: CLIP, BLIP, Stable Diffusion
- **Search & Retrieval**: FAISS, MS-COCO dataset
- **API & Frontend**: FastAPI, Gradio
- **Deployment**: Docker, Hugging Face Spaces
- **Performance**: DPMSolverMultistepScheduler, Hardware-specific optimizations

---


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
   
