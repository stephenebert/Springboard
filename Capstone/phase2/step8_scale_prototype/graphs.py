
"""
faiss_memory_latency_benchmark.py

Measure RSS memory, recall@10 and latency for different FAISS index types,
then plot Memory vs. Latency (point size/color ∝ Recall@10)
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

    # collect RSS after each build
    for name, recall10, latency in measurements:
        mem_mb = proc.memory_info().rss / 1024**2
        print(f"RSS after building {name}: {mem_mb:.1f} MB")

        results.append({
            "method":     name,
            "recall@10":  recall10,
            "latency_ms": latency,
            "memory_mb":  mem_mb,
        })

    df = pd.DataFrame(results)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8,6))

    scatter = sns.scatterplot(
        data=df,
        x="latency_ms", y="memory_mb",
        hue="recall@10", size="recall@10",
        palette="viridis", sizes=(50,300),
        edgecolor="w", linewidth=0.7,
        legend="brief"
    )


    for _, row in df.iterrows():
        plt.text(
            row.latency_ms + 0.03,  
            row.memory_mb + 0.3,    
            row.method,
            fontsize=9,
            ha="left", va="center"
        )

    plt.title("FAISS Index: Memory vs Latency\n(point size & color proportional to Recall@10)", fontsize=14)
    plt.xlabel("Avg query latency (ms)", fontsize=12)
    plt.ylabel("RAM footprint (MB)", fontsize=12)

    plt.legend(
        title="Recall@10",
        bbox_to_anchor=(1.02,1), loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
