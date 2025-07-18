# Image-to-Text Retrieval Demo  (BLIP → CLIP → FAISS)

Turn any image into a short caption with **BLIP**, embed that caption with **CLIP**, and retrieve the most similar human-written captions from MS-COCO using an in-memory **FAISS** index – all wrapped in a simple **Gradio** UI.

<p align="center">
  <img src="docs/demo_screenshot.png" width="720" alt="Screenshot of the Gradio demo"/>
</p>

---

## ✨ What It Does

1. **Upload an image**  
2. **BLIP** generates a caption  
3. **CLIP** encodes that caption to a 512-D embedding  
4. **FAISS** finds the *k* most similar captions from a pre-embedded COCO corpus  
5. Ranked results (distance ↓ = similarity ↑) are displayed

---

## Repository Layout


