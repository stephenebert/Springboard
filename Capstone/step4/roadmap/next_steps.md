# Roadmap & Next Steps

## Enhancements to Add
1. **Full‐caption evaluation**  
   - Count a retrieval as correct if the predicted caption matches *any* of the five ground-truth COCO captions.  
2. **Text→Image retrieval**  
   - Compute caption→image Recall@1 alongside image→text.  
3. **Larger-scale evaluation**  
   - Run both metrics on ≥1,000 randomly sampled COCO val images for stability.  
4. **Prompt-inversion integration**  
   - Attach the Croitoru *Reverse Stable Diffusion* inversion head to CLIP image embeddings.  
   - Evaluate prompt reconstruction accuracy on a held-out Stable Diffusion prompt dataset.  
5. **Round-trip retrieval**  
   - Pipeline: image → recover prompt → regenerate image.  
   - Measure reconstruction quality (e.g. perceptual similarity, CLIP score).

## Timeline & Milestones
- **Week 1:**  
  - Implement full-caption Recall@1  
  - Add text→image metric  
- **Week 2:**  
  - Scale both metrics to 1,000+ images  
  - Refine preprocessing & batching  
- **Week 3:**  
  - Integrate and test the inversion head  
  - Run inversion experiments & collect metrics  
- **Week 4:**  
  - Build round-trip demo script  
  - Quantify image reconstruction quality  
- **Week 5:**  
  - Write up results  
  - Polish notebooks, scripts, and presentations

## Deliverables
- **Notebooks**  
  - `notebooks/evaluate_full_metrics.ipynb`  
  - `notebooks/reproduce_inversion.ipynb`  
- **Scripts**  
  - `scripts/roundtrip_demo.py`  
- **Reports**  
  - `analysis/reproduction_analysis.md` (updated with full-caption & text→image results)  
  - Final Step 4 chapter in capstone write-up  
- **Presentation**  
  - 3–5 slide deck in `slides/step4_plan.pdf`

