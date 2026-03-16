"""
Read the CSV produced by sim_lambda_dist_compare.py and create a clear
2×2 subplot figure — one panel per idle timer, each comparing
exponential vs pareto simulation delay.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def main():
    # ---- locate CSV: optional CLI arg, else latest run from sweep_lambda_by_distribution ----
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        results_dir = Path("results_sim_dist_compare")
        if not results_dir.exists():
            print("Run sweep_lambda_by_distribution.py first, or pass CSV path as argument.")
            sys.exit(1)
        runs = sorted(results_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("No run_* directories in results_sim_dist_compare. Run sweep_lambda_by_distribution.py first.")
            sys.exit(1)
        csv_path = runs[0] / "sim_dist_compare_results.csv"
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    idle_timers = sorted(df["idle_timer"].unique())
    distributions = sorted(df["distribution"].unique())  # ['exponential', 'pareto']

    # ---- style map ----
    dist_style = {
        "exponential": {"color": "#1f77b4", "ls": "-",  "marker": "o", "label": "Exponential"},
        "pareto":      {"color": "#d62728", "ls": "--", "marker": "s", "label": "Pareto"},
    }

    # ---- 2×2 subplots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, T_idle in enumerate(idle_timers):
        ax = axes[idx]

        for dist_name in distributions:
            sub = df[(df["idle_timer"] == T_idle) & (df["distribution"] == dist_name)]
            s = dist_style[dist_name]

            ax.plot(
                sub["arrival_rate"],
                sub["sim_total_delay"],
                color=s["color"],
                linestyle=s["ls"],
                marker=s["marker"],
                markersize=3,
                linewidth=1.5,
                label=s["label"],
            )

        ax.set_title(rf"$\theta = {T_idle}$ s", fontsize=16)
        ax.set_yscale("log")
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=13, loc="upper left")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.6)

    # shared axis labels
    fig.supxlabel(r"$\lambda$, Arrival Rate (packets/s)", fontsize=16, y=0.02)
    fig.supylabel(r"$E[D]$, Average Delay (s)", fontsize=16, x=0.02)

    fig.tight_layout(rect=[0.03, 0.04, 1, 1])

    out_path = csv_path.parent / "sim_dist_compare_subplots.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
