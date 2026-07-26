"""Sweep arrival rate and plot analytical and simulated delay components."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from sim_v2 import run_single_configuration


# System parameters
TAU = 0.2
MU_SWITCH = 3.0
MU_CONTROLLER = 2.0
LAMBDAS = np.linspace(0.1, 2.99, 50)
IDLE_TIMEOUTS = (1.0, 2.0, 3.0, 4.0)

# Exponential arrivals and services; average ten independent replications.
SIM_TIME = 10_000.0
SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149)

COMPONENTS = {
    "total_delay": {"label": "Total delay", "color": "tab:blue"},
    "switch_delay": {"label": "Switch delay", "color": "tab:green"},
    "miss_delay": {"label": "Miss delay", "color": "tab:red"},
    "residual_delay": {"label": "Residual delay", "color": "tab:cyan"},
}


def analytical_delay_components(
    lambda_rate: float,
    mu_switch: float,
    mu_controller: float,
    tau: float,
    idle_timeout: float,
) -> dict[str, float]:
    """Return the analytical total delay and its three model components."""
    if lambda_rate >= mu_switch:
        return {component: np.nan for component in COMPONENTS}

    miss_probability = np.exp(-lambda_rate * idle_timeout)
    miss_rate = lambda_rate * miss_probability
    if miss_rate >= mu_controller:
        return {component: np.nan for component in COMPONENTS}

    controller_slack = mu_controller - miss_rate
    installation_mean = 2.0 * tau + 1.0 / controller_slack
    installation_second_moment = (
        4.0 * tau**2
        + 4.0 * tau / controller_slack
        + 2.0 / controller_slack**2
    )
    if miss_rate * installation_mean >= 1.0:
        return {component: np.nan for component in COMPONENTS}

    switch_delay = 1.0 / (mu_switch - lambda_rate)
    miss_delay = miss_probability * installation_mean
    residual_delay = miss_rate * installation_second_moment / 2.0
    return {
        "total_delay": switch_delay + miss_delay + residual_delay,
        "switch_delay": switch_delay,
        "miss_delay": miss_delay,
        "residual_delay": residual_delay,
    }


def run_sweep() -> pd.DataFrame:
    """Run ten replications for every timeout and arrival-rate pair."""
    rows = []
    total_runs = len(IDLE_TIMEOUTS) * len(LAMBDAS) * len(SEEDS)
    completed_runs = 0

    for idle_timeout in IDLE_TIMEOUTS:
        print(f"Running idle timeout Δ={idle_timeout:.1f}s")
        for lambda_rate in LAMBDAS:
            simulation_values = {component: [] for component in COMPONENTS}
            for seed in SEEDS:
                result = run_single_configuration(
                    lambda_rate=lambda_rate,
                    mu_switch=MU_SWITCH,
                    mu_controller=MU_CONTROLLER,
                    tau=TAU,
                    timeout=idle_timeout,
                    sim_time=SIM_TIME,
                    seed=seed,
                    dist="exponential",
                )
                stats = result["stats"]
                for component in COMPONENTS:
                    simulation_values[component].append(
                        stats.get(component, np.nan)
                    )
                completed_runs += 1

            analytical_values = analytical_delay_components(
                lambda_rate,
                MU_SWITCH,
                MU_CONTROLLER,
                TAU,
                idle_timeout,
            )
            row = {
                "idle_timeout": idle_timeout,
                "arrival_rate": lambda_rate,
                "replications": len(SEEDS),
                **{
                    f"simulation_{component}": float(
                        np.nanmean(values)
                    )
                    for component, values in simulation_values.items()
                },
                **{
                    f"analytical_{component}": value
                    for component, value in analytical_values.items()
                },
            }
            rows.append(row)
            print(
                f"\rCompleted {completed_runs}/{total_runs} simulations",
                end="",
                flush=True,
            )
        print()

    return pd.DataFrame(rows)


def plot_components(results: pd.DataFrame, output_path: Path):
    """Create four timeout panels with solids for theory and dots for simulation."""
    figure, axes = plt.subplots(
        1, len(IDLE_TIMEOUTS), figsize=(16, 4.4), sharey=True
    )

    for axis, idle_timeout in zip(axes, IDLE_TIMEOUTS):
        subset = results[results["idle_timeout"] == idle_timeout]
        for component, style in COMPONENTS.items():
            color = style["color"]
            axis.plot(
                subset["arrival_rate"],
                subset[f"analytical_{component}"],
                color=color,
                linewidth=1.8,
            )
            axis.plot(
                subset["arrival_rate"],
                subset[f"simulation_{component}"],
                color=color,
                marker="o",
                linestyle="none",
                markersize=3.5,
            )

        axis.set_title(rf"$\Delta={idle_timeout:.1f}\,\mathrm{{s}}$")
        axis.set_xlabel(r"Arrival rate $\lambda$ (packets/s)")
        axis.set_yscale("log")
        axis.grid(False)

    axes[0].set_ylabel("Average packet delay (s)")

    component_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linewidth=1.8,
            label=style["label"],
        )
        for style in COMPONENTS.values()
    ]
    representation_handles = [
        Line2D([0], [0], color="black", linewidth=1.8, label="Analytical"),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="none",
            markersize=4,
            label="Simulation mean",
        ),
    ]
    axes[0].legend(
        handles=component_handles + representation_handles,
        fontsize=8,
        loc="upper left",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    output_root = Path(__file__).resolve().parent / "results_delay_components"
    run_directory = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_directory.mkdir(parents=True, exist_ok=True)

    results = run_sweep()
    csv_path = run_directory / "delay_component_sweep.csv"
    png_path = run_directory / "delay_component_sweep.png"
    pdf_path = run_directory / "delay_component_sweep.pdf"

    results.to_csv(csv_path, index=False)
    plot_components(results, png_path)
    plot_components(results, pdf_path)

    print(f"CSV saved to : {csv_path}")
    print(f"PNG saved to : {png_path}")
    print(f"PDF saved to : {pdf_path}")


if __name__ == "__main__":
    main()
