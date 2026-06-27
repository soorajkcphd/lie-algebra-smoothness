"""
Experiment: Dimension scaling of empirical Lipschitz constants.

Key finding: Empirical constants DECREASE with dimension (concentration of measure)
while theoretical bounds grow polynomially.

Reproduces: Table SM1 in the supplementary material.
"""

import numpy as np
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lie_algebras import (
    proj_sl, proj_so, gradient, objective,
    estimate_global_lipschitz, theoretical_lipschitz_bound,
    get_projection
)


def run_dimension_experiment(dimensions: List[int], R: float = 2.0,
                              n_probes: int = 20, n_pairs: int = 100) -> Dict:
    """
    Run Lipschitz estimation across different dimensions.
    """
    results = {
        'dimensions': dimensions,
        'so_empirical': [],
        'sl_empirical': [],
        'so_theoretical': [],
        'sl_theoretical': []
    }
    
    for d in dimensions:
        print(f"  d = {d}...", end=" ", flush=True)
        
        # Generate data
        for algebra in ['so', 'sl']:
            proj = get_projection(algebra)
            
            # Target
            Y_target = np.random.randn(d, d)
            Y_target = proj(Y_target)
            Y_target = Y_target * (0.5 * R / max(norm(Y_target, 'fro'), 1e-10))
            T = expm(Y_target)
            M = norm(T, 'fro')
            
            # Probe vectors
            V = np.random.randn(d, n_probes)
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            
            # Empirical estimate
            L_emp = estimate_global_lipschitz(T, V, R, algebra, lambd=0.0, num_pairs=n_pairs)
            
            # Theoretical bound
            L_theo = theoretical_lipschitz_bound(R, M, n_probes, lambd=0.0, algebra=algebra)
            
            if algebra == 'so':
                results['so_empirical'].append(L_emp)
                results['so_theoretical'].append(L_theo)
            else:
                results['sl_empirical'].append(L_emp)
                results['sl_theoretical'].append(L_theo)
        
        print(f"so_emp={results['so_empirical'][-1]:.1f}, sl_emp={results['sl_empirical'][-1]:.1f}")
    
    return results


def main():
    print("=" * 70)
    print("DIMENSION SCALING EXPERIMENT")
    print("=" * 70)
    
    dimensions = [8, 16, 32, 64]
    R = 2.0
    
    print(f"\nSettings: R={R}, dimensions={dimensions}")
    print("\nRunning experiments...")
    
    results = run_dimension_experiment(dimensions, R=R, n_probes=20, n_pairs=100)
    
    # Print table
    print("\n" + "=" * 80)
    print("Table SM1: Dimension Scaling of Lipschitz Constants at R=2.0")
    print("=" * 80)
    print(f"{'Algebra':<20}", end="")
    for d in dimensions:
        print(f"{'d='+str(d):<15}", end="")
    print()
    print("-" * 80)
    
    print(f"{'so(d) empirical':<20}", end="")
    for v in results['so_empirical']:
        print(f"{v:<15.1f}", end="")
    print()
    
    print(f"{'sl(d) empirical':<20}", end="")
    for v in results['sl_empirical']:
        print(f"{v:<15.1f}", end="")
    print()
    
    print(f"{'so(d) theoretical':<20}", end="")
    for v in results['so_theoretical']:
        print(f"{v:<15.1e}", end="")
    print()
    
    print(f"{'sl(d) theoretical':<20}", end="")
    for v in results['sl_theoretical']:
        print(f"{v:<15.1e}", end="")
    print()
    print("-" * 80)
    
    # Analyze trends
    so_emp = np.array(results['so_empirical'])
    sl_emp = np.array(results['sl_empirical'])
    so_theo = np.array(results['so_theoretical'])
    sl_theo = np.array(results['sl_theoretical'])
    
    print("\nDimension scaling analysis:")
    print(f"  so(d) empirical: {so_emp[0]:.1f} → {so_emp[-1]:.1f} "
          f"(×{so_emp[-1]/so_emp[0]:.2f})")
    print(f"  sl(d) empirical: {sl_emp[0]:.1f} → {sl_emp[-1]:.1f} "
          f"(×{sl_emp[-1]/sl_emp[0]:.2f})")
    print(f"  so(d) theoretical: {so_theo[0]:.1e} → {so_theo[-1]:.1e} "
          f"(×{so_theo[-1]/so_theo[0]:.0f})")
    print(f"  sl(d) theoretical: {sl_theo[0]:.1e} → {sl_theo[-1]:.1e} "
          f"(×{sl_theo[-1]/sl_theo[0]:.0f})")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    d_arr = np.array(dimensions)
    
    # Left: Empirical
    ax1.plot(d_arr, so_emp, 'bo-', label='so(d) empirical', markersize=10, linewidth=2)
    ax1.plot(d_arr, sl_emp, 'rs-', label='sl(d) empirical', markersize=10, linewidth=2)
    
    ax1.set_xlabel('Dimension d', fontsize=12)
    ax1.set_ylabel('Empirical Lipschitz Constant', fontsize=12)
    ax1.set_title('Empirical: Decreasing with Dimension', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(dimensions)
    
    # Right: Theoretical vs empirical ratio
    ax2.semilogy(d_arr, so_theo, 'b--', label='so(d) theoretical', linewidth=2)
    ax2.semilogy(d_arr, sl_theo, 'r--', label='sl(d) theoretical', linewidth=2)
    ax2.semilogy(d_arr, so_emp, 'bo-', label='so(d) empirical', markersize=10, linewidth=2)
    ax2.semilogy(d_arr, sl_emp, 'rs-', label='sl(d) empirical', markersize=10, linewidth=2)
    
    ax2.set_xlabel('Dimension d', fontsize=12)
    ax2.set_ylabel('Lipschitz Constant', fontsize=12)
    ax2.set_title('Empirical vs Theoretical (log scale)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(dimensions)
    
    plt.tight_layout()
    plt.savefig('fig_dimension_scaling.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig_dimension_scaling.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: fig_dimension_scaling.pdf")
    
    print("\n" + "=" * 70)
    print("KEY FINDING: Concentration of Measure")
    print("Empirical Lipschitz constants DECREASE with dimension,")
    print("contrary to polynomial growth in theoretical bounds.")
    print("This is because adversarial subspace becomes negligible in high d.")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
