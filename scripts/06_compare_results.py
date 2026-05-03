"""Compare baseline and fine-tuned benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import ensure_dir


def load_metrics(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    metrics_dir = Path("results/metrics")
    baseline = load_metrics(metrics_dir / "baseline_metrics.json")
    finetuned = load_metrics(metrics_dir / "finetuned_metrics.json")

    rows = []
    for metric in sorted(set(baseline) | set(finetuned)):
        rows.append(
            {
                "metric": metric,
                "baseline": baseline.get(metric),
                "finetuned": finetuned.get(metric),
                "delta": (
                    finetuned.get(metric) - baseline.get(metric)
                    if isinstance(baseline.get(metric), (int, float))
                    and isinstance(finetuned.get(metric), (int, float))
                    else None
                ),
            }
        )

    output_dir = ensure_dir("results/reports")
    plots_dir = ensure_dir("results/plots")
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "benchmark_comparison.csv", index=False)

    plot_metrics = ["rouge1", "rouge2", "rougeL", "bleu"]
    plot_frame = table[table["metric"].isin(plot_metrics)].set_index("metric")
    plot_frame[["baseline", "finetuned"]].plot(kind="bar", figsize=(9, 5))
    plt.title("Base vs Fine-Tuned Model Text Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(plots_dir / "benchmark_comparison.png", dpi=160)

    print(table.to_string(index=False))
    print(f"Saved comparison table to {output_dir / 'benchmark_comparison.csv'}")
    print(f"Saved plot to {plots_dir / 'benchmark_comparison.png'}")


if __name__ == "__main__":
    main()
