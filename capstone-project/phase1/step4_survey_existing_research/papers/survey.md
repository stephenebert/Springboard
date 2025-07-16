# Survey of Existing Research

## Learning Transferable Visual Models From Natural Language Supervision  
**Alec Radford et al., 2021 (OpenAI)**  
**Problem & Motivation:** CLIP addresses the need for a zero-shot image–text alignment model that can generalize to unseen classes without any fine-tuning.  
**Dataset:** Trained on 400 million noisy image–text pairs scraped from the web.  
**Method:** Contrastive pretraining with two separate Transformer-based encoders (one for images, one for text) using a symmetric InfoNCE loss. Inference is zero-shot: both encoders are frozen and cosine similarity in embedding space is used for retrieval.  
**Results:**  
- Zero-shot ImageNet top-1 accuracy: 76.2%   
- MS COCO image→text Recall@1: 58.4%; text→image Recall@1: 37.8% :contentReference[oaicite:1]{index=1}  
**Links:** [PDF](https://arxiv.org/abs/2103.00020) | [GitHub](https://github.com/openai/CLIP)

---

## X-modaler: A Versatile and High-performance Codebase for Cross-modal Analytics  
**Yehao Li et al., 2021**  
**Problem & Motivation:** Provide an all-in-one toolkit for vision–language tasks—retrieval, captioning, VQA—built atop modern pretrained backbones like CLIP.  
**Dataset:** Benchmarks include MS COCO (retrieval & captioning), VQA v2.0, and others.  
**Method:** A unified PyTorch framework that wraps CLIP (or similar models) with task-specific heads and training scripts, offering end-to-end pipelines for data loading, training, and evaluation.  
**Results:** Matches or exceeds baseline performance on COCO retrieval (Recall@1 ≈ 55% image→text) and state-of-the-art captioning/VQA scores.  
**Links:** [PDF](https://arxiv.org/abs/2108.08217) | [GitHub](https://github.com/YehLi/xmodaler)

---

## Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval (HAT)  
**Yi Bin et al., 2023**  
**Problem & Motivation:** Most cross-modal retrieval systems use heterogeneous architectures (CNN for vision, Transformer/RNN for text), which can lead to mismatched embedding spaces.  
**Dataset:** Evaluated on MS COCO and Flickr30k retrieval benchmarks.  
**Method:** Proposes Hierarchical Alignment Transformers (HAT): identical Transformer backbones for both modalities plus a multi-level alignment module that fuses representations at different layers.  
**Results:**  
- MS COCO image→text Recall@1 improved by 7.6% relative; text→image Recall@1 by 16.7% relative.  
- Flickr30k image→text Recall@1 up 4.4%; text→image up 11.6% :contentReference[oaicite:2]{index=2}  
**Links:** [PDF](https://arxiv.org/abs/2308.04343) | [GitHub](https://github.com/LuminosityX/HAT)

---

## Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation  
**Daniel Csizmadia et al., 2025**  
**Problem & Motivation:** Further boost CLIP’s retrieval performance by distilling stronger cross-modal alignment into a compact student model.  
**Dataset:** Uses standard COCO and Flickr30k retrieval splits.  
**Method:** A teacher–student framework: CLIP (teacher) guides a cross-modal Transformer student via feature- and attention-map distillation losses, producing refined embeddings for both vision and text.  
**Results:**  
- COCO image→text Recall@1: 64.1% (vs. 58.4% CLIP); text→image R@1: 44.3%.  
- Flickr30k gains of +5–10% R@1 across both directions.  
**Links:** [PDF](https://arxiv.org/abs/2505.21549)  
Code & checkpoints: <https://anonymous.4open.science/r/DCLIP-B772/README.md>

---

## Reverse Stable Diffusion: What prompt was used to generate this image?  
**Florinel-Alin Croitoru et al., 2023**  
**Problem & Motivation:** Enable “prompt forensics” by recovering the original text prompt from a Stable Diffusion–generated image.  
**Dataset:** 200 K image–prompt pairs produced by Stable Diffusion v1.4 on diverse web prompts.  
**Method:** Freeze CLIP’s image encoder, train a small MLP regression head to predict CLIP text embeddings, and a classification head to reconstruct discrete prompt tokens.  
**Results:**  
- Prompt BLEU-2: 32.5  
- Cosine similarity between predicted and true text embeddings: 0.84   
**Links:** [PDF](https://arxiv.org/abs/2308.01472) | [GitHub](https://github.com/CroitoruAlin/Reverse-Stable-Diffusion)

---

## Image-to-Prompts (Jackson Chen)  
**Jackson Chen, 2023**  
**Problem & Motivation:** Provide an easy-to-use demo and library for inverting images to their generating prompts, leveraging publicly available SD datasets.  
**Dataset:** Uses the Kaggle “Image to Prompts” dataset of Stable Diffusion outputs and original prompts.  
**Method:** Extracts CLIP image embeddings, then feeds them into a transformer-based decoder to generate tokenized prompts. Includes Colab demos and a CLI.  
**Results:** Qualitative examples show high fidelity between generated and ground-truth prompts; quantitative metrics are not explicitly reported.  
**Links:** [GitHub](https://github.com/jacksonchen1998/Image-to-Prompts)

---

## Stable-Diffusion-Image-to-Prompts (Mingyuan Ren)  
**Mingyuan Ren, 2024**  
**Problem & Motivation:** Improve prompt recovery by jointly optimizing diffusion-model latents and a language model to decode text instructions from images.  
**Dataset:** COCO captions plus synthetically generated SD prompts.  
**Method:** Leverages the Stable Diffusion VAE encoder to extract latents, then optimizes a lightweight transformer-LM on top to predict discrete prompt tokens via cross-entropy loss.  
**Results:**  
- Exact prompt-token match accuracy: ~15% on held-out test set  
- BLEU-1 score of 45 on reconstructed prompts  
**Links:** [GitHub](https://github.com/MingyuanRen/Stable-Diffusion-Image-to-Prompts?utm_source=chatgpt.com)
