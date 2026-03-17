# Reactive SDN Simulation

Discrete-event simulation and analytical models for reactive SDN with idle flow timeout: packet delay vs arrival rate (λ), idle timeout Δ, and service/transmission parameters.

**Requirements:** Python 3.10+, `numpy`, `pandas`, `matplotlib`, `scipy` (for `optimal_timeout.py`).

---

## How to run

### Single run (one parameter set)

```bash
python sim.py [--lambda-rate 1.0] [--mu-switch 3.0] [--mu-controller 2.0] [--tau 0.4] [--timeout 2.0] [--sim-time 10000] [--seed 42] [--dist exponential]
# --timeout sets Δ (idle timeout)
```

Prints simulated mean delay, analytical mean delay, miss count, and total arrivals. Use `--dist` for inter-arrival/service distribution: `exponential`, `pareto`, `uniform`, `lognormal`.

### Sweeps and plots

| Command | Output |
|--------|--------|
| `python sweep_lambda.py` | Sweep over λ and Δ; CSV + PDF (log scale) → `results_sim_sweep/` |
| `python sweep_lambda_linear.py` | Same sweep; PDF with linear y-axis → `results_sim_sweep/` |
| `python sweep_timeout.py` | Sweep over Δ and λ; CSV + log/linear PDFs → `results_sim_sweep_timeout/` |
| `python sweep_lambda_by_distribution.py` | Compare exponential vs Pareto over λ; CSV + PDF → `results_sim_dist_compare/` |
| `python sweep_timeout_by_distribution.py` | Compare exponential vs Pareto over Δ; CSV + PDF → `results_sim_timeout_dist_compare/` |
| `python plot_distribution_comparison.py [path/to/results.csv]` | 2×2 subplot from distribution-comparison CSV (default: latest run in `results_sim_dist_compare/`) |

### Analytical-only and optimization

| Command | Description |
|--------|-------------|
| `python analytical_delay_components.py` | Analytical E[D] and components (switch, miss, residual) vs λ; CSV + PDF → `results_analytical/` |
| `python optimal_timeout.py` | Find optimal Δ* that minimizes E[D] (Newton + scipy); prints Δ* and E[D](Δ*). |

---

## Files

| File | Purpose |
|------|--------|
| **sim.py** | Core discrete-event simulation and analytical E[D] formula. Single-run CLI; also imported by all sweep scripts. |
| **sweep_lambda.py** | Sweep (λ × Δ); multiple seeds; writes CSV and log-scale delay plot. |
| **sweep_lambda_linear.py** | Same as above with linear y-axis and slightly different λ range. |
| **sweep_timeout.py** | Sweep (λ × Δ); delay vs Δ (log and linear PDFs). |
| **sweep_lambda_by_distribution.py** | Sweep λ for exponential and Pareto; compare sim vs analytical. |
| **sweep_timeout_by_distribution.py** | Sweep Δ for exponential and Pareto; compare sim vs analytical. |
| **plot_distribution_comparison.py** | Plot existing distribution-comparison CSV as 2×2 subplots (by Δ). Optional argument: path to CSV. |
| **analytical_delay_components.py** | Analytical model only: E[D] and components vs λ; no simulation. |
| **optimal_timeout.py** | Compute optimal Δ* (minimize E[D]) via Newton’s method and scipy. |

All sweep scripts create timestamped run directories (e.g. `results_sim_sweep/run_YYYYMMDD_HHMMSS_*`) and write CSV + PDF there.
