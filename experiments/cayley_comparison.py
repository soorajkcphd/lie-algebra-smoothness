"""
Experiment: Cayley vs Exponential retraction comparison on so(d).

Key finding: For compact algebras, both retractions have similar
bounded local smoothness (no exponential barrier).

NOTE ON THE CAYLEY DOMAIN (corrected):
For skew-symmetric X the eigenvalues of X are purely imaginary, so the
eigenvalues of (I - X/2) are 1 - i*theta/2 and have modulus >= 1.
Hence (I - X/2) is invertible for EVERY X in so(d), and Cay(X) is defined
on all of so(d) with no restriction on ||X||.  The previous version of this
script excluded samples with ||X||_2 >= 1.9 (and returned inf for ||X||_2 >= 2),
which is mathematically unjustified on so(d) and biased the Cayley averages by
filtering out exactly the large-norm samples that the exponential branch kept.
Those restrictions are removed below.

Reproduces: Figure 5 and Table 8.
"""

import numpy as np
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lie_algebras import proj_so, cayley, cayley_derivative


def objective_exp(X: np.ndarray, T: np.ndarray, V: np.ndarray) -> float:
    """Objective with exponential retraction."""
    G = expm(X)
    residual = G @ V - T @ V
    return 0.5 * np.sum(residual ** 2)


def objective_cayley(X: np.ndarray, T: np.ndarray, V: np.ndarray) -> float:
    """Objective with Cayley retraction.

    No domain restriction: on so(d), (I - X/2) is always invertible.
    """
    G = cayley(X)
    residual = G @ V - T @ V
    return 0.5 * np.sum(residual ** 2)


def gradient_exp(X: np.ndarray, T: np.ndarray, V: np.ndarray,
                 num_points: int = 30) -> np.ndarray:
    """Gradient with exponential retraction."""
    from lie_algebras import frechet_exp_adjoint

    G = expm(X)
    residual = G @ V - T @ V
    grad_f = residual @ V.T
    grad = frechet_exp_adjoint(X, grad_f, num_points)
    return proj_so(grad)


def gradient_cayley(X: np.ndarray, T: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Gradient with Cayley retraction.

    DCay(X)[H]  = (I - X/2)^{-1} H (I - X/2)^{-1}
    DCay(X)*[Z] = (I - X/2)^{-T} Z (I - X/2)^{-T}
    """
    d = X.shape[0]
    I = np.eye(d)

    G = cayley(X)
    residual = G @ V - T @ V          # d x n
    grad_f = residual @ V.T           # d x d

    inv = np.linalg.inv(I - X / 2)    # well-defined for all X in so(d)
    grad = inv.T @ grad_f @ inv.T

    return proj_so(grad)


def estimate_local_smoothness(X: np.ndarray, T: np.ndarray, V: np.ndarray,
                              retraction: str, num_directions: int = 20,
                              delta: float = 0.01) -> float:
    """Estimate local Lipschitz constant for the given retraction.

    Both retractions are evaluated on the SAME set of probe directions;
    no direction is skipped, so the two columns are directly comparable.
    """
    d = X.shape[0]

    if retraction == 'exp':
        grad_X = gradient_exp(X, T, V)
        grad_fn = lambda Y: gradient_exp(Y, T, V)
    else:  # cayley
        grad_X = gradient_cayley(X, T, V)
        grad_fn = lambda Y: gradient_cayley(Y, T, V)

    max_ratio = 0.0
    for _ in range(num_directions):
        U = np.random.randn(d, d)
        U = proj_so(U)
        U = U / norm(U, 'fro')

        Y = X + delta * U
        Y = proj_so(Y)

        grad_Y = grad_fn(Y)
        ratio = norm(grad_Y - grad_X, 'fro') / delta
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def run_comparison(d: int = 16, R_values: List[float] = None,
                   n_probes: int = 20, n_samples: int = 30,
                   seed: int = 0) -> Dict:
    """Compare exponential and Cayley retractions across radii.

    Every sampled X is used for BOTH retractions (no filtering), so the
    reported means are computed over identical sample sets.
    """
    rng = np.random.default_rng(seed)

    if R_values is None:
        R_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]

    # Target in SO(d)
    Y_target = proj_so(rng.standard_normal((d, d)))
    Y_target = Y_target * (0.5 / norm(Y_target, 'fro'))
    T = expm(Y_target)

    # Probe vectors (unit norm)
    V = rng.standard_normal((d, n_probes))
    V = V / np.linalg.norm(V, axis=0, keepdims=True)

    results = {
        'R_values': R_values,
        'exp_smoothness': [],
        'cayley_smoothness': [],
    }

    for R in R_values:
        print(f"  R = {R}...", end=" ", flush=True)

        exp_vals, cay_vals = [], []

        for _ in range(n_samples):
            X = proj_so(rng.standard_normal((d, d)))
            X = X * (R * rng.random() / max(norm(X, 'fro'), 1e-10))

            # Same X used for both retractions; nothing is skipped.
            exp_vals.append(estimate_local_smoothness(X, T, V, 'exp'))
            cay_vals.append(estimate_local_smoothness(X, T, V, 'cayley'))

        results['exp_smoothness'].append(float(np.mean(exp_vals)))
        results['cayley_smoothness'].append(float(np.mean(cay_vals)))

        print(f"exp={results['exp_smoothness'][-1]:.3f}, "
              f"cay={results['cayley_smoothness'][-1]:.3f}")

    return results


def main():
    print("=" * 70)
    print("CAYLEY VS EXPONENTIAL RETRACTION COMPARISON")
    print("=" * 70)

    d = 16
    R_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]

    print(f"\nSettings: d={d}, algebra=so(d) (compact)")
    print(f"R values: {R_values}")
    print("Cayley is defined on all of so(d): (I - X/2) is invertible for every")
    print("skew-symmetric X, since the eigenvalues of I - X/2 have modulus >= 1.")

    print("\nRunning comparison...")
    results = run_comparison(d, R_values, n_probes=20, n_samples=30, seed=0)

    # Table
    print("\n" + "=" * 70)
    print("Table 8: Cayley vs Exponential Local Smoothness on so(16)")
    print("=" * 70)
    print(f"{'R':<10}", end="")
    for R in R_values:
        print(f"{R:<12}", end="")
    print()
    print("-" * 70)

    print(f"{'Exp L_loc':<10}", end="")
    for v in results['exp_smoothness']:
        print(f"{v:<12.3f}", end="")
    print()

    print(f"{'Cay L_loc':<10}", end="")
    for v in results['cayley_smoothness']:
        print(f"{v:<12.3f}", end="")
    print()

    ratios = np.array(results['cayley_smoothness']) / np.array(results['exp_smoothness'])
    print(f"{'Ratio':<10}", end="")
    for r in ratios:
        print(f"{r:<12.3f}", end="")
    print()
    print("-" * 70)

    print(f"\nRatio statistics: mean={np.mean(ratios):.3f}, "
          f"range=[{np.min(ratios):.3f}, {np.max(ratios):.3f}]")

    # Figure  (no singularity line: none exists on so(d))
    fig, ax = plt.subplots(figsize=(10, 6))

    R = np.array(R_values)
    ax.plot(R, np.array(results['exp_smoothness']), 'bo-',
            label='Exponential', markersize=10, linewidth=2)
    ax.plot(R, np.array(results['cayley_smoothness']), 'rs-',
            label='Cayley', markersize=10, linewidth=2)

    ax.set_xlabel('Radius R (in Frobenius norm)', fontsize=12)
    ax.set_ylabel('Local Smoothness L_loc', fontsize=12)
    ax.set_title(f'Exponential vs Cayley Retraction on so({d})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.05)
    fig.tight_layout()

    plt.savefig('fig_cayley_comparison_ij.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig_cayley_comparison_ij.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: fig_cayley_comparison_ij.pdf")

    print("\n" + "=" * 70)
    print("KEY FINDING: On the compact algebra so(d), both retractions yield")
    print("similar bounded local smoothness. The exponential barrier is a")
    print("feature of NON-COMPACT algebras.")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
