"""
Experiment: Effect of sampling strategy on Lipschitz constant estimation.

This script compares random, boundary, and adversarial sampling strategies
for estimating the gradient Lipschitz constant.

Key finding: Adversarial sampling improves estimates by 10-25x.

Reproduces: Figure 1 and Table 1 in the main paper.
"""

import numpy as np
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lie_algebras import (
    proj_sl, proj_so, gradient, objective,
    hard_direction_sl, theoretical_lipschitz_bound,
    get_projection
)


def sample_random(d: int, R: float, algebra: str, n_samples: int) -> List[np.ndarray]:
    """Sample uniformly from interior of B_R(g)."""
    proj = get_projection(algebra)
    samples = []
    for _ in range(n_samples):
        X = np.random.randn(d, d)
        X = proj(X)
        r = R * np.random.rand() ** (1/d)  # Uniform in ball
        X = X * (r / max(norm(X, 'fro'), 1e-10))
        samples.append(X)
    return samples


def sample_boundary(d: int, R: float, algebra: str, n_samples: int) -> List[np.ndarray]:
    """Sample from boundary ||X||_F = R."""
    proj = get_projection(algebra)
    samples = []
    for _ in range(n_samples):
        X = np.random.randn(d, d)
        X = proj(X)
        X = X * (R / max(norm(X, 'fro'), 1e-10))
        samples.append(X)
    return samples


def sample_adversarial(d: int, R: float, algebra: str, n_samples: int) -> List[np.ndarray]:
    """Sample along hard direction with perturbations."""
    if algebra == 'sl':
        X0, mu = hard_direction_sl(d)
    else:
        X0 = np.zeros((d, d))
        X0[0, 0] = 1.0
        X0 = get_projection(algebra)(X0)
        X0 = X0 / max(norm(X0, 'fro'), 1e-10)
    
    proj = get_projection(algebra)
    samples = []
    
    for _ in range(n_samples):
        # Sample along hard direction with small perturbation
        t = R * (0.8 + 0.4 * np.random.rand())  # t ∈ [0.8R, 1.2R]
        X = t * X0
        
        # Add small random perturbation
        noise = np.random.randn(d, d) * 0.1
        noise = proj(noise)
        X = X + noise
        X = proj(X)
        
        # Project to ball
        if norm(X, 'fro') > R:
            X = X * (R / norm(X, 'fro'))
        
        samples.append(X)
    
    return samples


def estimate_lipschitz_from_samples(samples: List[np.ndarray], 
                                     T: np.ndarray, V: np.ndarray,
                                     algebra: str, lambd: float = 0.0) -> float:
    """Estimate Lipschitz constant from gradient differences on sample pairs."""
    n = len(samples)
    max_ratio = 0.0
    
    # Compute all gradients
    grads = [gradient(X, T, V, algebra, lambd, num_points=30) for X in samples]
    
    # Check all pairs
    for i in range(n):
        for j in range(i+1, n):
            diff_X = norm(samples[i] - samples[j], 'fro')
            if diff_X < 1e-10:
                continue
            diff_grad = norm(grads[i] - grads[j], 'fro')
            ratio = diff_grad / diff_X
            max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def run_experiment(d: int = 12, n_probes: int = 20, 
                   R_values: List[float] = [0.5, 1.0, 2.0, 3.0],
                   n_samples: int = 50) -> Dict:
    """
    Run the sampling comparison experiment.
    
    Returns dictionary with results for each sampling strategy.
    """
    algebra = 'sl'
    proj = get_projection(algebra)
    
    # Generate target and probe vectors
    Y_target = np.random.randn(d, d)
    Y_target = proj(Y_target)
    Y_target = Y_target * 0.5  # Moderate target
    T = expm(Y_target)
    M = norm(T, 'fro')
    
    V = np.random.randn(d, n_probes)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)  # Normalize columns
    
    results = {
        'R_values': R_values,
        'random': [],
        'boundary': [],
        'adversarial': [],
        'theoretical': []
    }
    
    for R in R_values:
        print(f"  R = {R}...", end=" ", flush=True)
        
        # Random sampling
        samples_rand = sample_random(d, R, algebra, n_samples)
        L_rand = estimate_lipschitz_from_samples(samples_rand, T, V, algebra)
        results['random'].append(L_rand)
        
        # Boundary sampling  
        samples_bnd = sample_boundary(d, R, algebra, n_samples)
        L_bnd = estimate_lipschitz_from_samples(samples_bnd, T, V, algebra)
        results['boundary'].append(L_bnd)
        
        # Adversarial sampling
        samples_adv = sample_adversarial(d, R, algebra, n_samples)
        L_adv = estimate_lipschitz_from_samples(samples_adv, T, V, algebra)
        results['adversarial'].append(L_adv)
        
        # Theoretical bound
        L_theo = theoretical_lipschitz_bound(R, M, n_probes, lambd=0.0, algebra=algebra)
        results['theoretical'].append(L_theo)
        
        print(f"rand={L_rand:.1f}, bnd={L_bnd:.1f}, adv={L_adv:.1f}, theo={L_theo:.1e}")
    
    return results


def main():
    print("=" * 70)
    print("SAMPLING STRATEGY COMPARISON")
    print("=" * 70)
    
    d = 12
    n_probes = 20
    R_values = [0.5, 1.0, 2.0, 3.0]
    
    print(f"\nSettings: d={d}, n_probes={n_probes}, algebra=sl(d)")
    print(f"Radii: {R_values}")
    print("\nRunning experiment...")
    
    results = run_experiment(d, n_probes, R_values, n_samples=50)
    
    # Print table
    print("\n" + "=" * 80)
    print("Table 1: Effect of Sampling Strategy on Empirical Lipschitz Constant")
    print("=" * 80)
    print(f"{'Strategy':<15} ", end="")
    for R in R_values:
        print(f"{'R='+str(R):<12}", end="")
    print(f"{'Ratio R=0.5':<15} {'Ratio R=3.0':<15}")
    print("-" * 80)
    
    for strategy in ['random', 'boundary', 'adversarial']:
        vals = results[strategy]
        theo = results['theoretical']
        print(f"{strategy.capitalize():<15} ", end="")
        for v in vals:
            print(f"{v:<12.2f}", end="")
        print(f"{vals[0]/theo[0]:<15.2e} {vals[-1]/theo[-1]:<15.2e}")
    
    print(f"{'Theoretical':<15} ", end="")
    for v in results['theoretical']:
        print(f"{v:<12.2e}", end="")
    print(f"{'1.0':<15} {'1.0':<15}")
    print("-" * 80)
    
    # Improvement factors
    print("\nImprovement (Adversarial / Random):")
    for i, R in enumerate(R_values):
        ratio = results['adversarial'][i] / max(results['random'][i], 1e-10)
        print(f"  R={R}: {ratio:.1f}x")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    R = np.array(R_values)
    
    # Left: Empirical Lipschitz constants
    ax1.semilogy(R, results['random'], 'bs-', label='Random', markersize=8, linewidth=2)
    ax1.semilogy(R, results['boundary'], 'ro-', label='Boundary', markersize=8, linewidth=2)
    ax1.semilogy(R, results['adversarial'], 'g^-', label='Adversarial', markersize=8, linewidth=2)
    ax1.semilogy(R, results['theoretical'], 'k--', label='Theoretical', linewidth=2)
    
    ax1.set_xlabel('Constraint Radius R', fontsize=12)
    ax1.set_ylabel('Lipschitz Constant Estimate', fontsize=12)
    ax1.set_title(f'Empirical vs Theoretical Lipschitz on sl({d})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Ratio to theoretical
    ratios_rand = np.array(results['random']) / np.array(results['theoretical'])
    ratios_bnd = np.array(results['boundary']) / np.array(results['theoretical'])
    ratios_adv = np.array(results['adversarial']) / np.array(results['theoretical'])
    
    ax2.semilogy(R, ratios_rand, 'bs-', label='Random', markersize=8, linewidth=2)
    ax2.semilogy(R, ratios_bnd, 'ro-', label='Boundary', markersize=8, linewidth=2)
    ax2.semilogy(R, ratios_adv, 'g^-', label='Adversarial', markersize=8, linewidth=2)
    
    ax2.set_xlabel('Constraint Radius R', fontsize=12)
    ax2.set_ylabel('Ratio (Empirical / Theoretical)', fontsize=12)
    ax2.set_title('Tightness of Bounds by Sampling Strategy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig1_adversarial_search.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig1_adversarial_search.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: fig1_adversarial_search.pdf")
    
    plt.show()


if __name__ == "__main__":
    main()
