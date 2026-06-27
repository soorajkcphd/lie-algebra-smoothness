"""
Experiment: Exact verification of lower bound construction.

This script validates Theorem 5.1 by measuring the Hessian norm along 
the hard direction X₀ = diag(1,-1,0,...,0)/√2 on sl(d).

Expected result: Fitted exponent = 2μ = √2 ≈ 1.4142

Reproduces: Figure 2 and Table 2 in the main paper.
"""

import numpy as np
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lie_algebras import (
    hard_direction_sl, proj_sl, objective, gradient,
    frechet_exp, hessian_exp_direction
)


def compute_hessian_norm_along_path(d: int, R_values: List[float], 
                                     num_points: int = 30) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ||∇²L(R·X₀)[X₀,X₀]|| for various R values.
    
    Args:
        d: Matrix dimension
        R_values: List of radius values to test
        num_points: Integration points
        
    Returns:
        empirical: Empirical Hessian norms
        theoretical: Theoretical predictions (μ²/2)e^{2μR}
        mu: The eigenvalue
    """
    X0, mu = hard_direction_sl(d)
    v = np.zeros(d)
    v[0] = 1.0  # Eigenvector of X0
    
    empirical = []
    theoretical = []
    
    for R in R_values:
        # Target: w = exp(R·X₀)v = e^{Rμ}v
        w = np.exp(R * mu) * v
        T = np.diag(np.exp(R * np.array([mu, -mu] + [0]*(d-2))))
        V = v.reshape(-1, 1)
        
        # Compute Hessian via finite differences on gradient
        X_R = R * X0
        eps = 1e-4
        
        grad_plus = gradient(X_R + eps * X0, T, V, 'sl', lambd=0.0, num_points=num_points)
        grad_minus = gradient(X_R - eps * X0, T, V, 'sl', lambd=0.0, num_points=num_points)
        
        # Directional second derivative: ∇²L[X₀,X₀] ≈ (∇L(X+εX₀) - ∇L(X-εX₀))·X₀ / (2ε)
        hess_dir = np.sum((grad_plus - grad_minus) * X0) / (2 * eps)
        empirical.append(abs(hess_dir))
        
        # Theoretical: (μ²/2)e^{2μR} from the Hessian formula
        # Note: The full second derivative at t=R is μ²e^{2μR}
        theo = (mu ** 2 / 2) * np.exp(2 * mu * R)
        theoretical.append(theo)
    
    return np.array(empirical), np.array(theoretical), mu


def fit_exponential(R_values: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = c * e^{αR} and return (α, c).
    """
    # Log-linear fit: log(y) = log(c) + αR
    log_values = np.log(values + 1e-10)
    coeffs = np.polyfit(R_values, log_values, 1)
    alpha = coeffs[0]
    c = np.exp(coeffs[1])
    return alpha, c


def main():
    print("=" * 70)
    print("LOWER BOUND VERIFICATION: Hard Direction on sl(d)")
    print("=" * 70)
    
    d = 8
    R_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    
    print(f"\nDimension d = {d}")
    print(f"Hard direction: X₀ = diag(1,-1,0,...,0)/√2")
    print(f"Eigenvalue μ = 1/√2 ≈ {1/np.sqrt(2):.4f}")
    print(f"Theoretical exponent: 2μ = √2 ≈ {np.sqrt(2):.4f}")
    
    # Compute Hessian norms
    print("\nComputing Hessian norms along hard direction...")
    empirical, theoretical, mu = compute_hessian_norm_along_path(d, R_values)
    
    # Fit exponential
    alpha_emp, c_emp = fit_exponential(R_values, empirical)
    alpha_theo, c_theo = fit_exponential(R_values, theoretical)
    
    # Compute ratios
    ratios = empirical / theoretical
    
    # Print results table
    print("\n" + "-" * 70)
    print(f"{'R':<8} {'Empirical':>12} {'Theory (μ²/2)e^2μR':>20} {'Ratio':>10}")
    print("-" * 70)
    for i, R in enumerate(R_values):
        print(f"{R:<8.1f} {empirical[i]:>12.2f} {theoretical[i]:>20.2f} {ratios[i]:>10.2f}")
    print("-" * 70)
    
    print(f"\nFitted exponents:")
    print(f"  Empirical:   α = {alpha_emp:.4f}")
    print(f"  Theoretical: α = {2*mu:.4f} (= 2μ = √2)")
    print(f"  Difference:  {abs(alpha_emp - 2*mu):.6f}")
    
    print(f"\nRatio (Emp/Theory): mean = {np.mean(ratios):.2f}, std = {np.std(ratios):.4f}")
    
    if abs(alpha_emp - np.sqrt(2)) < 0.01:
        print("\n✓ EXACT MATCH: Empirical exponent matches √2 to 2 decimal places!")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Log-scale plot
    ax1.semilogy(R_values, empirical, 'ro-', markersize=8, label='Empirical', linewidth=2)
    ax1.semilogy(R_values, theoretical, 'b--', label=f'Theory: (μ²/2)e^{{2μR}}', linewidth=2)
    
    # Add fitted line
    R_fine = np.linspace(R_values[0], R_values[-1], 100)
    fitted = c_emp * np.exp(alpha_emp * R_fine)
    ax1.semilogy(R_fine, fitted, 'g:', label=f'Fit: e^{{{alpha_emp:.4f}R}}', linewidth=2)
    
    ax1.set_xlabel('Radius R', fontsize=12)
    ax1.set_ylabel('Hessian norm ||∇²L[X₀,X₀]||', fontsize=12)
    ax1.set_title(f'Lower Bound Verification on sl({d})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Ratio plot
    ax2.plot(R_values, ratios, 'ko-', markersize=8, linewidth=2)
    ax2.axhline(y=np.mean(ratios), color='r', linestyle='--', 
                label=f'Mean ratio = {np.mean(ratios):.2f}')
    ax2.set_xlabel('Radius R', fontsize=12)
    ax2.set_ylabel('Ratio (Empirical / Theoretical)', fontsize=12)
    ax2.set_title('Constant Ratio Confirms Identical Scaling', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(ratios) * 1.5])
    
    plt.tight_layout()
    plt.savefig('fig2_hard_direction.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig2_hard_direction.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: fig2_hard_direction.pdf")
    
    plt.show()


if __name__ == "__main__":
    main()
