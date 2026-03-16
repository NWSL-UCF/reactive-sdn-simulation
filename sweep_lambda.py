import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from sim import run_single_configuration


# -----------------------------
# System parameters (match analytical_delay_components.py)
# -----------------------------
tau = 0.4    # one-way transmission delay (s)
mu_s = 3   # switch service rate (pkt/s)
mu_c = 2  # controller service rate (pkt/s)

# Arrival rates and idle timers (same combinations as analytical_delay_components.py)
lambdas = np.linspace(0.1, 2.99, 50)
idle_timers = [1.0, 2.0, 3.0, 4.0]

# Simulation settings
sim_time = 10_000.0
seeds = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]


def main():
    # -----------------------------
    # Set up output directory
    # -----------------------------
    output_root = Path("results_sim_sweep")
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "sdn_simulation_delay_results.csv"
    plot_path = run_dir / "simulation_delay_components.pdf"

    # -----------------------------
    # Run sim.py over all (lambda, idle_timer) combos
    # -----------------------------
    rows = []

    for T_idle in idle_timers:
        for lam_val in lambdas:
            seed_delays = []
            ana_delay = np.nan

            for s in seeds:
                result = run_single_configuration(
                    lambda_rate=lam_val,
                    mu_switch=mu_s,
                    mu_controller=mu_c,
                    tau=tau,
                    timeout=T_idle,
                    sim_time=sim_time,
                    seed=s,
                )
                stats = result["stats"]
                ana_delay = result["analytical_mean_delay"]
                seed_delays.append(stats.get("total_delay", np.nan))

            row = {
                "arrival_rate": lam_val,
                "idle_timer": T_idle,
                "sim_total_delay": float(np.nanmean(seed_delays)),
                "ana_mean_delay": ana_delay,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # Plotting: analytical_total_delay vs sim_total_delay
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["b", "r", "g", "m"]
    markers = ["o", "s", "^", "D"]

    for i, T_idle in enumerate(idle_timers):
        df_subset = df[df["idle_timer"] == T_idle]
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        label = f"Idle Timeout = {T_idle} s"

        ax.plot(
            df_subset["arrival_rate"],
            df_subset["sim_total_delay"],
            marker=m,
            linestyle="none",
            color=c,
            markersize=3,
            linewidth=1.0,
            label=label,
        )

        ax.plot(
            df_subset["arrival_rate"],
            df_subset["ana_mean_delay"],
            linestyle="-", # solid line for analytical delay
            color=c,
            linewidth=2.0,
        )

    sim_handle = plt.Line2D([], [], color="gray", marker="o", linestyle="-",
                            markersize=3, linewidth=1.0, label="Simulation")
    ana_handle = plt.Line2D([], [], color="gray", linestyle="-",
                            linewidth=2.0, label="Analytical")

    first_legend = ax.legend(loc="upper left", fontsize=14)
    ax.add_artist(first_legend)
    ax.legend(handles=[sim_handle, ana_handle], loc="lower right", fontsize=16)

    ax.set_xlabel(r"$\lambda$, Arrival Rate (packets/s)", fontsize=16)
    ax.set_ylabel(r"$E[D]$, Average Delay (s)", fontsize=16)
    ax.set_yscale("log")
    ax.tick_params(labelsize=16)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")

    print(f"Simulation sweep completed for {len(idle_timers)} idle_timers, {len(lambdas)} lambdas, {len(seeds)} seeds each.")
    print(f"CSV saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

