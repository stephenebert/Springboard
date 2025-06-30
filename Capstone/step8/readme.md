# Step 8: Scaling a Prototype

This step scales our cross-modal retrieval pipeline to the full training set (≈ 850 K samples) and benchmarks a variety of FAISS indices for large-scale, low-latency deployment. Step 8 brings our prototype from "research mode" to "production scale." In this phase, we embed the entire 850 K-sample corpus into high-dimensional feature vectors (images and captions) using a chunked, out-of-core pipeline that writes directly into HDF5. We then build and benchmark a family of FAISS indices exact (FlatIP) and approximate (IVF-Flat, IVF-PQ, HNSWFlat) to measure Recall@10, query latency, and RAM footprint at real-world scale. Finally, we sweep critical hyperparameters (nlist, nprobe, M, efSearch) to identify the optimal speed/accuracy trade-offs for deployment. By the end of Step 8, we’ll have a fully scalable retrieval backend tuned for sub-millisecond to low-millisecond performance with minimal memory overhead.

## High-Level Overview

**What "Step 8: Scale a Prototype" is doing:**

1. **Embed *everything*** - We take the *entire* training split of the combined COCO + Flickr-30k + Stable-Diffusion corpus (≈ 851 k image–caption pairs) and push them through our best backbone (ViT-B-32). *Output:* two giant 850,668 x 512 matrices—one for images, one for texts saved to `experiments/full/embeddings_full.h5`.

2. **Store them efficiently** - Instead of many small `.npy` files, we write a single HDF5 container with chunking + LZF compression, so we can memory-map slices without exhausting RAM.

3. **Build fast ANN indices (FAISS)** - We experiment with several approximate-nearest-neighbour layouts:
   * `FlatIP` (exact search, baseline)
   * `IVF-Flat` (coarse quantiser + exact residual)
   * `IVF-PQ` (coarse quantiser + product quantisation)
   * `HNSWFlat` (graph-based)
   
   For each index we measure **recall @ 10**, **average latency** (ms/query), and **RSS memory** right after building.

4. **Hyper-parameter sweeps** - We sweep `nlist / nprobe` for IVF and `M / efSearch` for HNSW to find the sweet spot where recall stays ≥ ~82% but latency and memory shrink dramatically.

5. **Cost / performance dashboards** - Everything is logged to CSV → piped into a quick `seaborn` scatter where:
   * **x-axis:** latency
   * **y-axis:** RAM
   * **marker size / colour:** recall
   
   That single plot lets reviewers eyeball the trade-offs instantly.

**Why?** Step 7 proved the concept on 10k samples; Step 8 proves it scales. One is left with:
* Embeddings one can ship to production.
* A FAISS index configuration that answers in ~1–2 ms with ~82% Recall@10.
* Concrete memory / latency / cost numbers for the README and slide-deck.

## Repository Structure
```
step8/
├── data/
│ └── metadata.parquet # Unified metadata from Step 7 (local)
├── experiments/ # (local; too large for git)
│ └── full/
│ ├── embeddings_full.h5 # HDF5: all image/text embeddings
│ ├── img_embs_full.npy
│ └── txt_embs_full.npy
├── scale_pipeline_hdf5.py # Chunk & embed to HDF5 (full dataset)
├── evaluate_retrieval.py # Compute exact Recall@K via brute-force
├── faiss_index_comparison.py # Compare FlatIP, IVF-Flat, HNSWFlat, IVF-PQ
├── faiss_param_sweep.py # Hyper-parameter sweep (nlist, nprobe, M, ef)
├── faiss_memory_latency_benchmark.py # RSS vs. latency vs. Recall plotter
├── graphs.py # Regenerate all figures
└── README.md 
```
---

## 1. Design decisions

| Component | Choice | Why it was chosen |
|-----------|--------|-------------------|
| **Embedding model** | OpenAI CLIP ViT-B/32 (512-d) | Strong open-source baseline; balances quality vs. time/VRAM for ≈1M samples. |
| **Storage format** | HDF5 (`embeddings_full.h5`) | Supports chunked writes/reads and compression; plays nicely with NumPy & FAISS. Parquet is better for tabular metadata but slower for large binary blobs. |
| **ANN engine** | FAISS IVF-Flat / IVF-PQ / HNSWFlat | Wide community adoption; CPU-only install keeps setup simple on basic hardware. IVF scales sub-linear; HNSW adds low-latency option. |
| **Chunk size** | 65,536 rows, 512 dims | Fits comfortably in <2GB RAM, enabling processing on 16GB laptops while saturating I/O. |
| **Benchmark metrics** | Recall@K, average latency, resident-set-size (RSS) | Mirror real-world retrieval QoS: accuracy, speed, and memory. |

> **Trade-offs:** IVF-Flat gives higher recall than IVF-PQ but at 4x memory; HNSW is even faster at K≤10 yet slower to build. The param-sweep script quantifies this and lets one pick a sweet spot for deployment geometry.
## 2. Quick-start

### 2.1  Installation

```bash
conda create -n capstone8 python=3.10 -y
conda activate capstone8
pip install -r requirements.txt
```

### 2.2 End-to-end run on the full dataset

**# 1 ) Encode images & captions --> HDF5**
```bash
python scale_pipeline_hdf5.py \
  --metadata_path data/metadata.parquet \
  --model clip-vit-base32 \
  --chunk_size 65536 \
  --output_path experiments/full/embeddings_full.h5
```

**# 2 ) Train an IVF-Flat index and benchmark it**
```bash
python index_benchmark.py \
  --embeddings_path experiments/full/embeddings_full.h5 \
  --index_type IVF4096,Flat \
  --topk 10 \
  --output_dir experiments/full/index_benchmarks
```

**# 3 ) Explore other hyper-parameters**
```bash
python faiss_param_sweep.py \
  --embeddings_path experiments/full/embeddings_full.h5 \
  --index_types IVF2048,PQ32 IVF4096,HNSW32 \
  --tops "5 10 30" \
  --out_csv experiments/full/param_sweep.csv
```


## 3. Chunked Embedding --> HDF5

In this first stage, we scale our embedding process to the full training set by streaming data in manageable chunks and writing the results into an HDF5 file. Using scale_pipeline_hdf5.py, we divide the ≈850 K images and captions into 2000-example blocks, embed each block with the ViT-B-32 model (pretrained on LAION-2B), and append the normalized 512-dim vectors to two HDF5 datasets (/image_embeddings and /text_embeddings). This approach lets us work around memory constraints and checkpoint progress safely. On our hardware, the end-to-end run took about 28 hours to complete, yielding a single embeddings_full.h5 file (≈1.4 GB) that contains all image/text vectors ready for large-scale indexing.
```bash
python scale_pipeline_hdf5.py \
  --parquet data/metadata.parquet \
  --out experiments/full/embeddings_full.h5 \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --device cpu \
  --batch 64 \
  --chunk 2000
```
Produces embeddings_full.h5 with two datasets:
- /image_embeddings (850668 x 512)
- /text_embeddings (850668 x 512)
  ![Figure 0: Embedding](Figure_0.png)
  - The embeddings took 28 hours to finish running.
## 4. Exact & Approximate Index Benchmark

In this benchmark we build an IVF-Flat index with 1024 centroids and then sweep the nprobe parameter (the number of lists visited at query time) from 1 up to 64. As shown in Figure 2, Recall@10 quickly reaches the exact search level (82.7 %) once we probe just 2 clusters and it never improves further by visiting more lists. Meanwhile, average query latency rises from 0.99 ms at nprobe=1 to 3.56 ms at nprobe=64. In other words, setting nprobe=2-4 gives us virtually the same retrieval accuracy as a full linear scan but with a 2-3x speedup, making it an excellent sweet spot for large-scale deployments.

```bash
python index_benchmark.py \
  --h5 experiments/full/embeddings_full.h5 \
  --nq 1000 \
  --k 10
```
![Figure 1: IVF-Flat (nlist=1024) | Sweep nprobe](Figure_1.png)
- Recall@10 saturates at 82.7 % for nprobe ≥ 2.
- Latency increases from 0.99 ms (nprobe=1) -> 3.56 ms (nprobe=64).
- Takeaway: nprobe = 2-4 offers near-optimal recall with 2-3x speedup over exact.

## 5. Compare Multiple ANN Indices

In this comparison we pit four different FAISS index types against one another FlatIP (exact inner‐product), IVF-Flat (nlist=1024, nprobe=16), IVF-PQ (nlist=1024, m=64), and HNSWFlat (M=32) to see how they trade off accuracy and speed (Figure 3). As expected, the exact FlatIP search tops out at 82.7 % Recall@10 but has the highest latency (~5.3 ms/query). Both IVF-Flat and IVF-PQ match that recall level: IVF-Flat does so in about 2.3 ms, while IVF-PQ achieves it in only ~1.0 ms nearly a 5x speedup over the exact scan. HNSWFlat, on the other hand, delivers sub 0.1 ms queries but sacrifices recall (80.6 %). All four indices consume roughly the same RAM footprint (~152 MB), so the choice really comes down to latency‐versus‐accuracy needs: IVF-PQ offers the best of both worlds (exact recall at 5x lower latency), IVF-Flat gives a 2x speedup with zero loss in accuracy, and HNSWFlat is ultra‐fast but at a steep cost in recall.
```bash
python faiss_index_comparison.py \
  --h5 experiments/full/embeddings_full.h5 \
  --nq 1000 \
  --k 10
```
![Figure 2: Approximate Index Comparison](Figure_2.png)
| Index                            | Recall@10 | Latency (ms) |
|----------------------------------|-----------|--------------|
| FlatIP (exact inner-product)     | 82.7 %    | 5.28         |
| IVF-Flat (nlist=1024,nprobe=16)  | 82.7 %    | 2.30         |
| IVF-PQ (nlist=1024,m=64)         | 82.7 %    | 1.03         |
| HNSWFlat (M=32)                  | 80.6 %    | 0.02         |

**Note:** IVF-PQ matches exact recall at ~5x lower latency.

## 6. Memory vs. Latency Trade-off

This figure shows that all four FAISS indices occupy roughly the same modest RAM footprint (around 152 MB), so memory use is not the primary concern. Latency versus accuracy is. At one extreme, FlatIP (exact search) delivers the highest recall (82.7 %) but at the cost of a relatively slow ∼4.1 ms per query. Moving left on the latency axis, IVF-Flat cuts that time in half (∼1.8 ms) without sacrificing accuracy, and IVF-PQ further accelerates queries to under 1 ms while still matching FlatIP’s 82.7 % recall—achieving a nearly 5x speedup with no loss in performance. HNSWFlat pushes latency down into the tens of microseconds but at a notable drop in recall (∼80.6 %). Overall, IVF-PQ stands out as the best trade-off for large-scale deployment, combining exact-search accuracy with sub-millisecond speed at minimal memory overhead.
```bash
python faiss_memory_latency_benchmark.py
```
![Figure 3: RAM Footprint vs. Latency (point size/color proportional to Recall@10)](Figure_3.png)
- RAM footprint ≈ 152 MB for all indices.
- Latency spans 0.02 ms --> 5.28 ms per query.
- Bubble size & color map to Recall@10, highlighting trade-offs:
  -- IVF-PQ & IVF-Flat: best accuracy & moderate speed.
  -- HNSWFlat: ultra-fast but lower recall.
  
## 7. Hyperparameter Sweeps

To fine-tune our approximate search indices, we ran an automated sweep over each index's key knobs: for IVF-Flat, the number of Voronoi cells (nlist) and the number of cells probed at query time (nprobe); and for HNSWFlat, both the graph connectivity (M) and the search depth (efSearch). By systematically measuring recall@10, query latency, and memory usage across these settings, we found that IVF-Flat with nlist=1024, nprobe=16 consistently matched the exact-search recall of 82.7 % while halving the average query time, and that IVF-PQ offered similar accuracy with sub-millisecond lookups. These results confirm that either IVF-Flat (nlist=1024, nprobe=16) or IVF-PQ (nlist=1024, m=64) are the best choices for production, delivering near-optimal accuracy at dramatically reduced cost.
```bash
python faiss_param_sweep.py \
  --h5 experiments/full/embeddings_full.h5 \
  --nq 1000 \
  --k 10
```
- Choice for production: IVF-Flat(nlist=1024,nprobe=16) or IVF-PQ.

-  Sweeps over nlist, nprobe, M, efSearch.

  
## 8. Index Benchmarking
Finally, we built and compared four FAISS index types FlatIP (exact inner-product), IVF-Flat (ANN with Voronoi cells), HNSWFlat (graph-based), and IVF-PQ (product quantization) on the full 850 K embedding dataset. For each, we measured:

- Recall@10 to quantify accuracy against the brute-force baseline,

- Average latency to assess real-time query performance, and

- RAM footprint (RSS) to gauge memory overhead.

Our benchmarks show that exact FlatIP yields the highest recall (82.7 %) but suffers a 4-5 ms/query latency. IVF-Flat (nlist=1024, nprobe=16) matches that recall with ∼1.7 ms average latency, a 2-3x speedup. IVF-PQ further boosts throughput to <1 ms/query without loss in recall, offering a nearly 5x speedup. HNSWFlat (M=32, efSearch=32) delivers ultra-low latency (<0.03 ms) but at the cost of a significant recall drop (∼80.6 %). All indices consume roughly 1.5 GB of RAM. Overall, this comprehensive benchmarking highlights that IVF-Flat and IVF-PQ strike the best balance of accuracy, speed, and memory efficiency for large-scale deployment.



Build various FAISS indices and measure their performance based on:
- Recall@10 (exact vs. ANN)
- Avg. latency (ms/query)
- RAM footprint (RSS in MB)

python index_benchmark.py \
  --h5 experiments/full/embeddings_full.h5 \
  --k 10 \
  --nlist 1024 \
  --nprobe_list 1 2 4 8 16 32 64 \
  --hnsw_m_list 8 16 32 \
  --hnsw_ef_list 8 16 32 64 128
  
  **Sample output:**
  
| Index Type            | R@10    | Latency (ms) | RSS (MB) |
|-----------------------|---------|--------------|----------|
| FlatIP                | 82.70%  | 4.56         | 1702     |
| IVF-Flat(nlist=1024,nprobe=16) | 82.70%  | 1.17         | 5087     |
| HNSWFlat(M=32, ef=32) | 81.10%  | 0.03         | 2802     |
| ...                   | ...     | ...          | ...      |
---



# Key Findings
- Exact (FlatIP): Highest recall (82.7 %) but slowest (≈ 5 ms/query).

- IVF-Flat (nlist=1024, nprobe=16): Same recall, 2-3x faster, minimal RAM.

- IVF-PQ: Matches recall & latency (~1 ms) at very low memory.

- HNSWFlat (M=32): Sub-ms queries but recall drops to ~80 %.

- Recommendation: Use IVF-Flat(nlist=1024, nprobe=16) in production for optimal speed-accuracy trade-off.

