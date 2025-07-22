# Analysis of CLIP Baseline Reproduction

## 1. Comparison to Published Results
- **Recall@1:** 0.81 (evaluated on 100 COCO val images, matching only the first caption)  
- **Paper’s Recall@1:** ~0.58 (evaluated on the full COCO val set, with any of the five captions considered a match)

## 2. Possible Reasons for Discrepancy
- **Single-caption evaluation:** Using only the first caption per image inflates matches versus allowing any of the five.  
- **Sample size:** 100 images is a small subset; on a larger sample the metric would likely drop.  
- **Preprocessing/tokenization:** Differences in image resizing, text tokenization, or batching compared to the paper’s setup.

## 3. Strengths
- **High retrieval accuracy** on this small subset demonstrates the end-to-end pipeline works.  
- **Simplicity:** Only a few lines of code needed to reproduce core CLIP behavior.  
- **Speed:** Runs in seconds on 100 images, making iterative experimentation easy.

## 4. Limitations
- **Overestimation risk:** Single-caption matching and small sample size can give an overly optimistic score.  
- **One-way retrieval:** We haven’t yet measured text→image performance.  
- **Limited qualitative evaluation:** We reviewed only a handful of examples; broader qualitative analysis could reveal failure modes.

## 5. Takeaways & Next Steps
- **Evaluate against all captions:** Update the metric to count a match if the predicted index corresponds to *any* of the image’s five ground-truth captions.  
- **Add text→image retrieval:** Compute and report recall for caption→image direction as well.  
- **Scale up sample size:** Run on at least 1,000 randomly sampled images for more stable estimates.  
- **Integrate prompt inversion:** Once retrieval is solid, move on to attaching the inversion head.
