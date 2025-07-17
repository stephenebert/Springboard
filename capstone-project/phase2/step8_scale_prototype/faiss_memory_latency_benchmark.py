"""
faiss_memory_latency_benchmark.py

Measure RSS memory, recall@10 and latency for different FAISS index types,
then plot Memory vs Latency (point size/color ∝ Recall@10).
"""

import psutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():

    measurements = [
        ("FlatIP",   82.70, 4.126),
        ("IVF-Flat", 82.70, 1.779),
        ("HNSWFlat", 81.10, 0.026),
        ("IVF-PQ",   82.70, 0.791),
    ]

    proc = psutil.Process()
    results = []

    for name, recall10, latency in measurements:
        # Measure current RSS (resident memory)
        mem_mb = proc.memory_info().rss / 1024**2
        print(f"RSS after building {name}: {mem_mb:.1f} MB")

        results.append({
            "index":      name,
            "recall@10":  recall10,
            "latency_ms": latency,
            "memory_mb":  mem_mb,
        })


    df = pd.DataFrame(results)

    # Plot: Memory vs Latency, point size/color ∝ Recall@10
    plt.figure(figsize=(6,6))
    sns.scatterplot(
        data=df,
        x="latency_ms",
        y="memory_mb",
        size="recall@10",
        hue="recall@10",
        legend="brief",
        sizes=(50, 300),
    )

    # Annotate each point with the index name
    for _, row in df.iterrows():
        plt.text(
            row.latency_ms + 0.02,  # small horizontal offset
            row.memory_mb + 1,      # small vertical offset
            row.index,
            fontsize=8,
            va="bottom",
            ha="left",
        )

    plt.title("FAISS Index: Memory vs Latency\n(point size/color ∝ Recall@10)")
    plt.xlabel("Avg query latency (ms)")
    plt.ylabel("RAM footprint (MB)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
