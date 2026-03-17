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

# Single arrival rate
lam = 1.0  # pkt/s

# Idle timeout sweep
idle_timers = np.linspace(1.0, 5.0, 50)

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
    output_root = Path("results_sim_timeout_dist_compare")
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "sim_timeout_dist_compare_results.csv"
    plot_path = run_dir / "sim_timeout_dist_compare.pdf"

    # -----------------------------
    # Run simulation for each distribution × idle_timer
    # -----------------------------
    rows = []

    for dist_cfg in distributions:
        dist_name = dist_cfg["name"]
        print(f"Running distribution: {dist_name} ...")

        for T_idle in idle_timers:
            seed_delays = []

            for s in seeds:
                result = run_single_configuration(
                    lambda_rate=lam,
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
                "idle_timer": T_idle,
                "sim_total_delay": float(np.nanmean(seed_delays)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # Compute analytical curve (dense for smooth line)
    # -----------------------------
    timers_dense = np.linspace(1.0, 5.0, 500)
    ana_delays = [
        analytical_mean_delay(lam, mu_s, mu_c, tau, t)
        for t in timers_dense
    ]

    # -----------------------------
    # Plotting: exponential markers, pareto markers, analytical line
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    dist_style = {
        "exponential": {"color": "blue", "marker": "o", "label": "Simulation (Poisson)"},
        "pareto":      {"color": "red", "marker": "s", "label": "Simulation (Pareto)"},
    }

    for dist_cfg in distributions:
        dist_name = dist_cfg["name"]
        s = dist_style[dist_name]
        df_sub = df[df["distribution"] == dist_name]

        ax.plot(
            df_sub["idle_timer"],
            df_sub["sim_total_delay"],
            marker=s["marker"],
            linestyle="none",
            color=s["color"],
            markersize=5,
            label=s["label"],
        )

    # --- Analytical curve (continuous solid line) ---
    ax.plot(
        timers_dense,
        ana_delays,
        linestyle="-",
        color="green",
        linewidth=2.0,
        label="Analytical",
    )

    ax.legend(fontsize=16, loc="upper right")
    ax.set_xlabel(r"$\theta$, Idle Timeout (s)", fontsize=16)
    ax.set_ylabel(r"$E[D]$, Average Delay (s)", fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")

    print(f"\nSweep completed: {len(distributions)} distributions, "
          f"{len(idle_timers)} timeouts, {len(seeds)} seeds each.")
    print(f"λ = {lam} pkt/s")
    print(f"CSV saved to : {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
