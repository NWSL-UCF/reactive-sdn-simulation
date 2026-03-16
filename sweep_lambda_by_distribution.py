import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from sim import run_single_configuration, analytical_mean_delay


# -----------------------------
# System parameters
# -----------------------------
tau = 0.4    # one-way transmission delay (s)
mu_s = 3     # switch service rate (pkt/s)
mu_c = 2     # controller service rate (pkt/s)

# Arrival rates and single idle timer
lambdas = np.linspace(0.1, 2.99, 50)
T_idle = 3.0  # single idle timeout (s)

# Distributions to compare
distributions = [
    {"name": "exponential", "dist": "exponential", "dist_shape": 3.0},
    {"name": "pareto",      "dist": "pareto",      "dist_shape": 3.0},
]

# Simulation settings
sim_time = 10_000.0
seeds = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]


def main():
    # -----------------------------
    # Set up output directory
    # -----------------------------
    output_root = Path("results_sim_dist_compare")
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "sim_dist_compare_results.csv"
    plot_path = run_dir / "sim_dist_compare.pdf"

    # -----------------------------
    # Run simulation for each distribution × lambda
    # -----------------------------
    rows = []

    for dist_cfg in distributions:
        dist_name = dist_cfg["name"]
        print(f"Running distribution: {dist_name} ...")

        for lam_val in lambdas:
            seed_delays = []

            for s in seeds:
                result = run_single_configuration(
                    lambda_rate=lam_val,
                    mu_switch=mu_s,
                    mu_controller=mu_c,
                    tau=tau,
                    timeout=T_idle,
                    sim_time=sim_time,
                    seed=s,
                    dist=dist_cfg["dist"],
                    dist_shape=dist_cfg["dist_shape"],
                )
                stats = result["stats"]
                seed_delays.append(stats.get("total_delay", np.nan))

            row = {
                "distribution": dist_name,
                "arrival_rate": lam_val,
                "idle_timer": T_idle,
                "sim_total_delay": float(np.nanmean(seed_delays)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # Compute analytical curve (dense for smooth line)
    # -----------------------------
    lambdas_dense = np.linspace(0.1, 2.99, 500)
    ana_delays = [
        analytical_mean_delay(lam, mu_s, mu_c, tau, T_idle)
        for lam in lambdas_dense
    ]

    # -----------------------------
    # Plotting: exponential markers, pareto markers, analytical line
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Simulation data points (markers only, no connecting line) ---
    dist_style = {
        "exponential": {"color": "#1f77b4", "marker": "o", "label": "Sim (Exponential)"},
        "pareto":      {"color": "#d62728", "marker": "s", "label": "Sim (Pareto)"},
    }

    for dist_cfg in distributions:
        dist_name = dist_cfg["name"]
        s = dist_style[dist_name]
        df_sub = df[df["distribution"] == dist_name]

        ax.plot(
            df_sub["arrival_rate"],
            df_sub["sim_total_delay"],
            marker=s["marker"],
            linestyle="none",
            color=s["color"],
            markersize=5,
            label=s["label"],
        )

    # --- Analytical curve (continuous solid line) ---
    ax.plot(
        lambdas_dense,
        ana_delays,
        linestyle="-",
        color="black",
        linewidth=2.0,
        label="Analytical",
    )

    ax.legend(fontsize=16, loc="upper left")
    ax.set_xlabel(r"$\lambda$, Arrival Rate (packets/s)", fontsize=16)
    ax.set_ylabel(r"$E[D]$, Average Delay (s)", fontsize=16)
    ax.set_yscale("log")
    ax.tick_params(labelsize=16)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")

    print(f"\nSweep completed: {len(distributions)} distributions, "
          f"{len(lambdas)} lambdas, {len(seeds)} seeds each.")
    print(f"Idle timeout: {T_idle} s")
    print(f"CSV saved to : {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
