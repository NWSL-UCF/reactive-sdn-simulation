"""
Find optimal idle timeout using Newton's method to minimize E[D](Δ).

Parameters: λ = 1, μ_s = 3, μ_c = 2, τ = 0.4
"""

import numpy as np
import scipy.optimize


def mean_delay(delta, lam, mu_s, mu_c, tau):
    """
    Compute E[D](Δ) from Eq. (ED_expanded).
    
    E[D] = 1/(μ_s - λ) + e^(-λΔ) * (2τ + 1/α) 
           + (λ e^(-λΔ)/2) * (4τ² + 4τ/α + 2/α²)
    where α = μ_c - λ e^(-λΔ)
    """
    W_s = 1.0 / (mu_s - lam)
    alpha = mu_c - lam * np.exp(-lam * delta)
    
    if alpha <= 0:
        return np.inf  # Unstable
    
    term1 = W_s
    term2 = np.exp(-lam * delta) * (2 * tau + 1.0 / alpha)
    term3 = (lam * np.exp(-lam * delta) / 2.0) * (
        4 * tau**2 + 4 * tau / alpha + 2.0 / (alpha**2)
    )
    
    return term1 + term2 + term3


def mean_delay_derivative(delta, lam, mu_s, mu_c, tau):
    """
    Compute dE[D]/dΔ using chain rule.
    
    d/dΔ [e^(-λΔ) * (2τ + 1/α)] = -λ e^(-λΔ) * (2τ + 1/α) 
                                   + e^(-λΔ) * (-λ² e^(-λΔ) / α²)
    
    d/dΔ [(λ e^(-λΔ)/2) * (4τ² + 4τ/α + 2/α²)] = 
        -λ² e^(-λΔ)/2 * (4τ² + 4τ/α + 2/α²)
        + (λ e^(-λΔ)/2) * (-4τ λ² e^(-λΔ) / α² - 4λ² e^(-λΔ) / α³)
    """
    alpha = mu_c - lam * np.exp(-lam * delta)
    
    if alpha <= 0:
        return np.nan  # Unstable
    
    exp_term = np.exp(-lam * delta)
    dalpha_ddelta = lam**2 * exp_term
    
    # Derivative of term2: e^(-λΔ) * (2τ + 1/α)
    dterm2 = -lam * exp_term * (2 * tau + 1.0 / alpha) + exp_term * (
        -dalpha_ddelta / (alpha**2)
    )
    
    # Derivative of term3: (λ e^(-λΔ)/2) * (4τ² + 4τ/α + 2/α²)
    term3_base = 4 * tau**2 + 4 * tau / alpha + 2.0 / (alpha**2)
    dterm3_base = -4 * tau * dalpha_ddelta / (alpha**2) - 4 * dalpha_ddelta / (alpha**3)
    
    dterm3 = -lam**2 * exp_term / 2.0 * term3_base + (
        lam * exp_term / 2.0
    ) * dterm3_base
    
    return dterm2 + dterm3


def theoretical_min_delay(lam, mu_s):
    """Theoretical minimum E[D] as Δ → ∞: 1/(μ_s - λ)."""
    return 1.0 / (mu_s - lam)


def find_optimal_timeout_newton(
    lam, mu_s, mu_c, tau,
    T=10.0,
    x_pct=1.0,
    initial_guess=2.0,
    tol=1e-8,
    max_iter=100,
):
    """
    Find optimal Δ* using Newton's method.
    Bounds: Δ ∈ [0.1, T].
    Early stop: return when E[D] ≤ E[D]_min * (1 + x_pct/100).
    """
    E_D_min = theoretical_min_delay(lam, mu_s)
    E_D_target = E_D_min * (1.0 + x_pct / 100.0)
    delta = initial_guess

    for i in range(max_iter):
        E_D = mean_delay(delta, lam, mu_s, mu_c, tau)
        if E_D <= E_D_target:
            return delta  # Within x% of theoretical minimum

        f_val = mean_delay_derivative(delta, lam, mu_s, mu_c, tau)

        if np.abs(f_val) < tol:
            break

        h = 1e-6
        f_prime = (
            mean_delay_derivative(delta + h, lam, mu_s, mu_c, tau)
            - mean_delay_derivative(delta - h, lam, mu_s, mu_c, tau)
        ) / (2 * h)

        if np.abs(f_prime) < 1e-10:
            break

        delta_new = delta - f_val / f_prime

        if delta_new < 0.1:
            delta_new = 0.1
        if delta_new > T:
            delta_new = T

        if np.abs(delta_new - delta) < tol:
            delta = delta_new
            break

        delta = delta_new

    return delta


def find_optimal_timeout_scipy(lam, mu_s, mu_c, tau, T=10.0):
    """
    Find optimal Δ* using scipy.optimize.minimize.
    Bounds: Δ ∈ [0.1, T]. Serves as verification.
    """
    result = scipy.optimize.minimize_scalar(
        lambda d: mean_delay(d, lam, mu_s, mu_c, tau),
        bounds=(0.1, T),
        method='bounded',
    )
    return result.x


def find_minimum_idle_timeout(lam, mu_s, mu_c, tau, T=10.0, x_pct=1.0, tol=1e-8):
    """
    Find minimum idle timeout Δ ∈ [0.1, T] such that E[D] ≤ E[D]_min * (1 + x_pct/100).
    Returns the smallest Δ achieving delay within x% of theoretical minimum.
    """
    E_D_min = theoretical_min_delay(lam, mu_s)
    E_D_target = E_D_min * (1.0 + x_pct / 100.0)

    lo, hi = 0.1, T
    if mean_delay(hi, lam, mu_s, mu_c, tau) > E_D_target:
        return hi  # Even at T we don't reach target; return T

    for _ in range(200):  # ~200 iterations gives ~1e-60 precision
        mid = (lo + hi) / 2.0
        if mean_delay(mid, lam, mu_s, mu_c, tau) <= E_D_target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# Alias for backward compatibility
find_optimal_timeout_target = find_minimum_idle_timeout


if __name__ == "__main__":
    # System parameters
    lam = 1.0
    mu_s = 3.0
    mu_c = 2.0
    tau = 0.4
    T = 100.0       # Upper bound on Δ (s)
    x_pct = 0.01   # Stop when E[D] within x% of theoretical minimum

    E_D_min = theoretical_min_delay(lam, mu_s)
    E_D_target = E_D_min * (1.0 + x_pct / 100.0)

    print("=" * 60)
    print("Finding Optimal Idle Timeout")
    print("=" * 60)
    print(f"Parameters: λ = {lam}, μ_s = {mu_s}, μ_c = {mu_c}, τ = {tau} s")
    print(f"Bounds: Δ ∈ [0.1, {T}] s")
    print(f"Target: E[D] ≤ {E_D_target:.6f} s ({x_pct}% above theoretical min {E_D_min:.6f} s)")
    print()

    # Primary result: Minimum idle timeout for x%
    print("Minimum Idle Timeout (within x% of theoretical min)")
    print("-" * 60)
    delta_min = find_minimum_idle_timeout(lam, mu_s, mu_c, tau, T=T, x_pct=x_pct)
    E_D_at_min = mean_delay(delta_min, lam, mu_s, mu_c, tau)
    print(f"Minimum Δ* = {delta_min:.6f} s  (achieves E[D] within {x_pct}% of {E_D_min:.6f} s)")
    print(f"E[D] = {E_D_at_min:.6f} s")
    print()

    # Newton's method (early stop when within target)
    print("Newton's Method (verification)")
    print("-" * 60)
    delta_opt_newton = find_optimal_timeout_newton(lam, mu_s, mu_c, tau, T=T, x_pct=x_pct)
    E_D_opt_newton = mean_delay(delta_opt_newton, lam, mu_s, mu_c, tau)
    print(f"Δ* = {delta_opt_newton:.6f} s, E[D] = {E_D_opt_newton:.6f} s")
    print()

    # Scipy (verification, minimizes over [0.1, T])
    print("Scipy minimize_scalar (verification)")
    print("-" * 60)
    delta_opt_scipy = find_optimal_timeout_scipy(lam, mu_s, mu_c, tau, T=T)
    E_D_opt_scipy = mean_delay(delta_opt_scipy, lam, mu_s, mu_c, tau)
    print(f"Optimal Δ* = {delta_opt_scipy:.6f} s")
    print(f"E[D] = {E_D_opt_scipy:.6f} s")
    print()

    # Compare some values around the optimum
    print("E[D] for various Δ values:")
    print("-" * 60)
    print(f"{'Δ (s)':>8} | {'E[D] (s)':>12}")
    print("-" * 60)
    for delta_test in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        E_D_test = mean_delay(delta_test, lam, mu_s, mu_c, tau)
        marker = " <-- min" if abs(delta_test - delta_min) < 0.1 else ""
        print(f"{delta_test:8.2f} | {E_D_test:12.6f}{marker}")

    print()
    print("=" * 60)
