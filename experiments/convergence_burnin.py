"""
Experiment: Convergence analysis with burn-in period.

This script shows that using global worst-case step-sizes leads to
very slow convergence, especially for non-compact algebras.

Reproduces: Figure SM1 in the supplementary material.
"""

import numpy as np
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lie_algebras import (
    proj_sl, proj_so, gradient, objective,
    theoretical_lipschitz_bound, get_projection, proj_ball
)


def run_pgd(algebra: str, d: int, R: float, n_probes: int,
            n_iters: int, use_global_stepsize: bool = True) -> Dict:
    """
    Run projected gradient descent and track convergence.
    """
    proj = get_projection(algebra)
    
    # Generate target
    Y_target = np.random.randn(d, d)
    Y_target = proj(Y_target)
    Y_target = Y_target * (0.5 * R / norm(Y_target, 'fro'))
    T = expm(Y_target)
    M = norm(T, 'fro')
    
    # Probe vectors
    V = np.random.randn(d, n_probes)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    
    # Compute step size
    L_global = theoretical_lipschitz_bound(R, M, n_probes, lambd=0.0, algebra=algebra)
    
    if use_global_stepsize:
        step_size = 0.5 / L_global
    else:
        step_size = 0.01  # Fixed reasonable step size
    
    # Initialize
    X = np.random.randn(d, d) * 0.1
    X = proj(X)
    
    # Track progress
    losses = [objective(X, T, V, 0.0)]
    grad_norms = []
    min_grad_norms = []  # Running minimum
    
    min_grad_sq = float('inf')
    
    for t in range(n_iters):
        grad = gradient(X, T, V, algebra, lambd=0.0, num_points=30)
        grad_norm_sq = norm(grad, 'fro') ** 2
        grad_norms.append(grad_norm_sq)
        
        min_grad_sq = min(min_grad_sq, grad_norm_sq)
        min_grad_norms.append(min_grad_sq)
        
        # Update
        X_new = X - step_size * grad
        X_new = proj(X_new)
        X_new = proj_ball(X_new, R)
        X = X_new
        
        losses.append(objective(X, T, V, 0.0))
    
    return {
        'losses': np.array(losses),
        'grad_norms_sq': np.array(grad_norms),
        'min_grad_norms_sq': np.array(min_grad_norms),
        'L_global': L_global,
        'step_size': step_size,
        'algebra': algebra,
        'd': d,
        'R': R
    }


def fit_convergence_rate(iterations: np.ndarray, values: np.ndarray, 
                         burn_in: int = 100) -> float:
    """Fit convergence rate after burn-in period."""
    # Use iterations after burn-in
    mask = iterations >= burn_in
    if np.sum(mask) < 10:
        return 0.0
    
    log_iter = np.log(iterations[mask])
    log_vals = np.log(values[mask] + 1e-15)
    
    # Linear fit
    coeffs = np.polyfit(log_iter, log_vals, 1)
    return coeffs[0]  # Slope


def main():
    print("=" * 70)
    print("CONVERGENCE ANALYSIS WITH BURN-IN")
    print("=" * 70)
    
    d = 16
    R = 2.0
    n_probes = 20
    n_iters = 2000
    
    print(f"\nSettings: d={d}, R={R}, n_probes={n_probes}")
    
    # Run for so(d) - compact
    print("\nRunning so(d) (compact)...")
    results_so = run_pgd('so', d, R, n_probes, n_iters)
    
    # Run for sl(d) - non-compact
    print("Running sl(d) (non-compact)...")
    results_sl = run_pgd('sl', d, R, n_probes, n_iters)
    
    # Print statistics
    print("\n" + "-" * 50)
    print(f"{'Algebra':<10} {'L_global':<15} {'Step size':<15}")
    print("-" * 50)
    print(f"{'so('+str(d)+')':<10} {results_so['L_global']:<15.2e} {results_so['step_size']:<15.2e}")
    print(f"{'sl('+str(d)+')':<10} {results_sl['L_global']:<15.2e} {results_sl['step_size']:<15.2e}")
    print("-" * 50)
    
    # Fit convergence rates
    burn_in = 200
    iterations = np.arange(1, n_iters + 1)
    
    rate_so = fit_convergence_rate(iterations, results_so['min_grad_norms_sq'], burn_in)
    rate_sl = fit_convergence_rate(iterations, results_sl['min_grad_norms_sq'], burn_in)
    
    print(f"\nConvergence rates (after burn-in at t={burn_in}):")
    print(f"  so({d}): slope = {rate_so:.3f} (theory: -1.0)")
    print(f"  sl({d}): slope = {rate_sl:.3f} (theory: -1.0)")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top row: so(d)
    ax1, ax2 = axes[0]
    
    ax1.semilogy(results_so['losses'], 'b-', linewidth=1)
    ax1.axvline(x=burn_in, color='r', linestyle='--', label=f'Burn-in ({burn_in})')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss L(X)', fontsize=12)
    ax1.set_title(f'so({d}) (Compact): Loss vs Iteration', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.loglog(iterations, results_so['grad_norms_sq'], 'b-', alpha=0.5, label='||∇L||²')
    ax2.loglog(iterations, results_so['min_grad_norms_sq'], 'orange', linewidth=2, 
               label='min_{s≤t} ||∇L||²')
    
    # Add theoretical O(1/T) line
    t_range = np.logspace(np.log10(burn_in), np.log10(n_iters), 50)
    c = results_so['min_grad_norms_sq'][burn_in] * burn_in
    ax2.loglog(t_range, c / t_range, 'gray', linestyle='--', label='O(1/T) (slope=-1)')
    
    ax2.axvline(x=burn_in, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Iteration (log scale)', fontsize=12)
    ax2.set_ylabel('||∇L||² (log scale)', fontsize=12)
    ax2.set_title(f'so({d}): Gradient Norm (fitted slope: {rate_so:.3f})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom row: sl(d)
    ax3, ax4 = axes[1]
    
    ax3.semilogy(results_sl['losses'], 'b-', linewidth=1)
    ax3.axvline(x=burn_in, color='r', linestyle='--', label=f'Burn-in ({burn_in})')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Loss L(X)', fontsize=12)
    ax3.set_title(f'sl({d}) (Non-Compact): Loss vs Iteration', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.loglog(iterations, results_sl['grad_norms_sq'], 'b-', alpha=0.5, label='||∇L||²')
    ax4.loglog(iterations, results_sl['min_grad_norms_sq'], 'orange', linewidth=2,
               label='min_{s≤t} ||∇L||²')
    
    c = results_sl['min_grad_norms_sq'][burn_in] * burn_in
    ax4.loglog(t_range, c / t_range, 'gray', linestyle='--', label='O(1/T) (slope=-1)')
    
    ax4.axvline(x=burn_in, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Iteration (log scale)', fontsize=12)
    ax4.set_ylabel('||∇L||² (log scale)', fontsize=12)
    ax4.set_title(f'sl({d}): Gradient Norm (fitted slope: {rate_sl:.3f})', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig4_convergence_burnin.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig4_convergence_burnin.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: fig4_convergence_burnin.pdf")
    
    print("\n" + "=" * 70)
    print("KEY FINDING: Global step-sizes are overly conservative")
    print(f"  so({d}) achieves slope {rate_so:.3f} (vs theoretical -1.0)")
    print(f"  sl({d}) achieves slope {rate_sl:.3f} (vs theoretical -1.0)")
    print("This validates the need for adaptive/local smoothness methods.")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
