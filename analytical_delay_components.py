import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

# -----------------------------
# System parameters
# -----------------------------
tau = 0.4     # one-way transmission delay (s)
mu_s = 3.0     # switch service rate (pkt/s)
mu_c = 2.0     # controller service rate (pkt/s)

# Arrival rates
lambdas = np.linspace(0.1, 3.0, 50)
# print(lambdas)

# Idle timers to compare
idle_timers = [1]

# Set up output directory (portable)
output_root = Path("results_analytical")
run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
run_dir.mkdir(parents=True, exist_ok=True)

csv_path = run_dir / "sdn_detailed_delay_results.csv"
plot_path = run_dir / "average_packet_delay_components.pdf"
parameters_path = run_dir / "paramters.txt"

parameters_text = "\n".join([
    "tau = 0.4     # one-way transmission delay (s)",
    "mu_s = 3.0     # switch service rate (pkt/s)",
    "mu_c = 2.0     # controller service rate (pkt/s)",
    "",
    "# Arrival rates",
    "lambdas = np.linspace(0.1, 3.0, 200)",
    "# print(lambdas)",
    "",
    "# Idle timers to compare",
    "idle_timers = [1, 2, 3, 4, 5]",
])
parameters_path.write_text(parameters_text, encoding="utf-8")


def average_delay(lam, idle_timer):
    """
    Average packet delay for reactive SDN model.
    Returns a dictionary of calculated values or NaNs if stability conditions are violated.
    """
    p = np.exp(-lam * idle_timer)   # miss probability
    lambda_m = p * lam              # miss arrival rate

    # Stability conditions
    if lam >= mu_s or lambda_m >= mu_c:
        return {
            'p': p, 'lambda_m': lambda_m, 'alpha': np.nan, 'inv_alpha': np.nan,
            'E_T': np.nan, 'E_T2': np.nan, 'switch_delay': np.nan,
            'miss_delay': np.nan, 'residual_delay': np.nan, 'average_delay': np.nan
        }

    alpha = mu_c - lambda_m
    inv_alpha = 1 / alpha if alpha != 0 else np.nan

    # Mean and second moment of installation time
    E_T = 2 * tau + inv_alpha
    E_T2 = 4 * tau**2 + 4 * tau * inv_alpha + 2 * inv_alpha**2

    # Installation utilization constraint
    if lambda_m * E_T >= 1:
        return {
            'p': p, 'lambda_m': lambda_m, 'alpha': alpha, 'inv_alpha': inv_alpha,
            'E_T': E_T, 'E_T2': E_T2, 'switch_delay': np.nan,
            'miss_delay': np.nan, 'residual_delay': np.nan, 'average_delay': np.nan
        }

    # Individual delay components
    switch_delay = 1 / (mu_s - lam)
    miss_delay = p * E_T
    residual_delay = lambda_m * E_T2 / 2
    total_average_delay = switch_delay + miss_delay + residual_delay

    return {
        'p': p,
        'lambda_m': lambda_m,
        'alpha': alpha,
        'inv_alpha': inv_alpha,
        'E_T': E_T,
        'E_T2': E_T2,
        'switch_delay': switch_delay,
        'miss_delay': miss_delay,
        'residual_delay': residual_delay,
        'average_delay': total_average_delay
    }


# -----------------------------
# Data collection for CSV and Plotting
# -----------------------------
if __name__ == "__main__":
    results_data = []

    for T_idle in idle_timers:
        for lam_val in lambdas:
            delay_metrics = average_delay(lam_val, T_idle)

            # Collect data for CSV, including all returned metrics
            row = {'arrival_rate': lam_val, 'idle_timer': T_idle}
            row.update(delay_metrics)
            results_data.append(row)

    # Create DataFrame and save to CSV
    df_detailed_results = pd.DataFrame(results_data)
    df_detailed_results.to_csv(csv_path, index=False)

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['b', 'g', 'r', 'c', 'm']

    for i, T_idle in enumerate(idle_timers):
        df_subset = df_detailed_results[df_detailed_results['idle_timer'] == T_idle]

        ax.plot(df_subset['arrival_rate'], df_subset['average_delay'],
                label=f'Total Delay (T_idle={T_idle}s)',
                marker=markers[i], linestyle='-', color=colors[i], markersize=5)

        ax.plot(df_subset['arrival_rate'], df_subset['switch_delay'],
                label=f'Switch Delay (T_idle={T_idle}s)',
                marker='x', linestyle='-', color=colors[i + 1], markersize=5)

        ax.plot(df_subset['arrival_rate'], df_subset['miss_delay'],
                label=f'Miss Delay (T_idle={T_idle}s)',
                marker='+', linestyle='-', color=colors[i + 2], markersize=5)

        ax.plot(df_subset['arrival_rate'], df_subset['residual_delay'],
                label=f'Residual Delay (T_idle={T_idle}s)',
                marker='^', linestyle='-.', color=colors[i + 3], markersize=5)

    ax.set_xlabel("Arrival rate (packets/s)")
    ax.set_ylabel("Average packet delay (seconds)")
    ax.set_title("Average Packet Delay and Components vs Arrival Rate")
    ax.set_yscale("log")
    ax.legend(loc='upper left')
    ax.grid(True, which="both", ls="-")
    fig.tight_layout()
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.show()

    print(f"CSV saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")
    print(f"Parameters saved to: {parameters_path}")

