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


def find_optimal_timeout_newton(lam, mu_s, mu_c, tau, initial_guess=2.0, tol=1e-8, max_iter=100):
    """
    Find optimal Δ* using Newton's method.
    Solves dE[D]/dΔ = 0.
    """
    delta = initial_guess
    
    for i in range(max_iter):
        f_val = mean_delay_derivative(delta, lam, mu_s, mu_c, tau)
        
        if np.abs(f_val) < tol:
            break
        
        # Compute second derivative for Newton's method
        # Use finite difference approximation
        h = 1e-6
        f_prime = (
            mean_delay_derivative(delta + h, lam, mu_s, mu_c, tau)
            - mean_delay_derivative(delta - h, lam, mu_s, mu_c, tau)
        ) / (2 * h)
        
        if np.abs(f_prime) < 1e-10:
            break
        
        delta_new = delta - f_val / f_prime
        
        # Ensure delta stays positive and reasonable
        if delta_new < 0.1:
            delta_new = 0.1
        if delta_new > 10.0:
            delta_new = 10.0
        
        if np.abs(delta_new - delta) < tol:
            delta = delta_new
            break
        
        delta = delta_new
    
    return delta


def find_optimal_timeout_scipy(lam, mu_s, mu_c, tau, bounds=(0.1, 10.0)):
    """
    Find optimal Δ* using scipy.optimize.minimize.
    This serves as a verification.
    """
    result = scipy.optimize.minimize_scalar(
        lambda d: mean_delay(d, lam, mu_s, mu_c, tau),
        bounds=bounds,
        method='bounded',
    )
    return result.x


if __name__ == "__main__":
    # System parameters
    lam = 1.0
    mu_s = 3.0
    mu_c = 2.0
    tau = 0.4
    
    print("=" * 60)
    print("Finding Optimal Idle Timeout")
    print("=" * 60)
    print(f"Parameters: λ = {lam}, μ_s = {mu_s}, μ_c = {mu_c}, τ = {tau} s")
    print()
    
    # Method 1: Newton's method
    print("Method 1: Newton's Method")
    print("-" * 60)
    delta_opt_newton = find_optimal_timeout_newton(lam, mu_s, mu_c, tau)
    E_D_opt_newton = mean_delay(delta_opt_newton, lam, mu_s, mu_c, tau)
    print(f"Optimal Δ* = {delta_opt_newton:.6f} s")
    print(f"Minimum E[D] = {E_D_opt_newton:.6f} s")
    print(f"Derivative at optimum: {mean_delay_derivative(delta_opt_newton, lam, mu_s, mu_c, tau):.2e}")
    print()
    
    # Method 2: Scipy (verification)
    print("Method 2: Scipy minimize_scalar (verification)")
    print("-" * 60)
    delta_opt_scipy = find_optimal_timeout_scipy(lam, mu_s, mu_c, tau)
    E_D_opt_scipy = mean_delay(delta_opt_scipy, lam, mu_s, mu_c, tau)
    print(f"Optimal Δ* = {delta_opt_scipy:.6f} s")
    print(f"Minimum E[D] = {E_D_opt_scipy:.6f} s")
    print()
    
    # Compare some values around the optimum
    print("E[D] for various Δ values:")
    print("-" * 60)
    print(f"{'Δ (s)':>8} | {'E[D] (s)':>12}")
    print("-" * 60)
    for delta_test in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        E_D_test = mean_delay(delta_test, lam, mu_s, mu_c, tau)
        marker = " <-- optimal" if abs(delta_test - delta_opt_newton) < 0.1 else ""
        print(f"{delta_test:8.2f} | {E_D_test:12.6f}{marker}")
    
    print()
    print("=" * 60)
