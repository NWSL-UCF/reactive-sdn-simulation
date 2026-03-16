import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from sim import run_single_configuration


# -----------------------------
# System parameters
# -----------------------------
tau = 0.4    # one-way transmission delay (s)
mu_s = 3     # switch service rate (pkt/s)
mu_c = 2     # controller service rate (pkt/s)

# Idle timeout sweep
idle_timers = np.linspace(1.0, 5.0, 50)

# Arrival rates to compare
lambda_rates = [0.10, 0.33, 0.66, 1.0]

# Simulation settings
sim_time = 10_000.0
seeds = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]


def main():
    # -----------------------------
    # Set up output directory
    # -----------------------------
    output_root = Path("results_sim_sweep_timeout")
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "sdn_simulation_delay_results.csv"
    plot_log_path = run_dir / "delay_vs_timeout_log.pdf"
    plot_linear_path = run_dir / "delay_vs_timeout_linear.pdf"

    # -----------------------------
    # Run sim over all (lambda, idle_timer) combos
    # -----------------------------
    rows = []

    for lam_val in lambda_rates:
        for T_idle in idle_timers:
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
                "lambda_rate": lam_val,
                "idle_timer": T_idle,
                "sim_total_delay": float(np.nanmean(seed_delays)),
                "ana_mean_delay": ana_delay,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # Plotting helper
    # -----------------------------
    colors = ["b", "r", "g", "m"]
    markers = ["o", "s", "^", "D"]

    def make_plot(ax, use_log):
        for i, lam_val in enumerate(lambda_rates):
            df_subset = df[df["lambda_rate"] == lam_val]
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            label = rf"$\lambda$ = {lam_val} pkt/s"

            ax.plot(
                df_subset["idle_timer"],
                df_subset["sim_total_delay"],
                marker=m,
                linestyle="none",
                color=c,
                markersize=3,
                linewidth=1.0,
                label=label,
            )

            ax.plot(
                df_subset["idle_timer"],
                df_subset["ana_mean_delay"],
                linestyle="-",
                color=c,
                linewidth=2.0,
            )

        sim_handle = plt.Line2D([], [], color="gray", marker="o", linestyle="-",
                                markersize=3, linewidth=1.0, label="Simulation")
        ana_handle = plt.Line2D([], [], color="gray", linestyle="-",
                                linewidth=1.8, label="Analytical")

        first_legend = ax.legend(loc="lower left", fontsize=16)
        ax.add_artist(first_legend)
        ax.legend(handles=[sim_handle, ana_handle], loc="upper right", fontsize=16)

        ax.set_xlabel(r"$\Delta$, Idle Timeout (s)", fontsize=16)
        ax.set_ylabel(r"$E[D]$, Average Delay (s)", fontsize=16)
        if use_log:
            ax.set_yscale("log")
        ax.tick_params(labelsize=16)
        ax.grid(False)

    # Log-scale plot
    fig_log, ax_log = plt.subplots(figsize=(8, 8))
    make_plot(ax_log, use_log=True)
    fig_log.tight_layout()
    fig_log.savefig(plot_log_path, format="pdf", bbox_inches="tight")

    # Linear-scale plot
    fig_lin, ax_lin = plt.subplots(figsize=(8, 8))
    make_plot(ax_lin, use_log=False)
    fig_lin.tight_layout()
    fig_lin.savefig(plot_linear_path, format="pdf", bbox_inches="tight")

    print(f"Sweep completed for {len(lambda_rates)} lambdas, {len(idle_timers)} timeouts, {len(seeds)} seeds each.")
    print(f"CSV saved to: {csv_path}")
    print(f"Log plot saved to: {plot_log_path}")
    print(f"Linear plot saved to: {plot_linear_path}")


if __name__ == "__main__":
    main()
