# lie-algebra-smoothness

# Intrinsic Smoothness Barriers for Matrix Exponential Optimization on Lie Algebras

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code to reproduce all experiments from the paper:

> Intrinsic Smoothness Barriers for Optimization via the Matrix Exponential on Lie Algebras
> Sooraj K.C, Vivek Mishra
> Indian Journal of Pure and Applied Mathematics (IJPAM), 2026

## Overview

We study the gradient Lipschitz constant (smoothness) for optimization problems of the form

    min_{X in g} L(exp(X))

where g is a matrix Lie algebra. Our main results establish:

1. Upper bounds: L_grad <= C(n, M, R) * exp(2R) for general Lie algebras.
2. Lower bounds: L_grad >= c(mu) * exp(2|mu|R) for non-compact algebras.
3. Compact / non-compact dichotomy: radius-independent smoothness for so(d), su(d)
   versus exponential growth for sl(d), gl(d), sp(2k).

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- JAX (optional, for automatic differentiation)

### Setup

    # Clone the repository
    git clone https://github.com/soorajkcphd/lie-algebra-smoothness.git
    cd lie-algebra-smoothness

    # Create virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt

    # Install package in development mode
    pip install -e .

## Repository Structure

    lie-algebra-smoothness/
    |-- README.md
    |-- LICENSE
    |-- requirements.txt
    |-- setup.py
    |-- pyproject.toml
    |
    |-- src/
    |   |-- lie_smoothness/
    |       |-- __init__.py
    |       |-- algebras.py          # Lie algebra definitions (so, sl, gl, sp, su)
    |       |-- exponential.py       # Matrix exponential and Frechet derivatives
    |       |-- objectives.py        # Optimization objectives
    |       |-- smoothness.py        # Lipschitz constant estimation
    |       |-- optimization.py      # Projected gradient descent
    |       |-- utils.py             # Helper functions
    |
    |-- experiments/
    |   |-- run_all.py               # Run all experiments
    |   |-- adversarial_sampling.py  # Figure 1, Table 2
    |   |-- hard_direction.py        # Figure 2, Table 3
    |   |-- local_smoothness.py      # Figure 3
    |   |-- convergence_burnin.py    # Figure 4
    |   |-- dimension_scaling.py     # Dimension-scaling table
    |   |-- cayley_comparison.py     # Cayley comparison figure and table
    |
    |-- tests/
    |   |-- test_algebras.py
    |   |-- test_exponential.py
    |   |-- test_smoothness.py
    |
    |-- figures/                     # Generated figures
    |   |-- fig1_adversarial_search_ij.pdf
    |   |-- fig2_hard_direction_ij.pdf
    |   |-- fig3_local_smoothness_ij.pdf
    |   |-- fig4_convergence_burnin_ij.pdf
    |   |-- fig_cayley_comparison_ij.pdf
    |
    |-- results/                     # Numerical results (CSV)
        |-- ...

## Quick Start

### Basic Usage

    import numpy as np
    from lie_smoothness import algebras, exponential, smoothness

    # Define a Lie algebra
    d = 8
    algebra = algebras.SpecialLinear(d)   # sl(d)

    # Generate a random element
    X = algebra.random_element(radius=2.0)

    # Compute matrix exponential and its Frechet derivative
    G = exponential.expm(X)
    dG = exponential.frechet_derivative(X, direction=algebra.random_element(radius=0.1))

    # Estimate local smoothness
    L_loc = smoothness.estimate_local(X, algebra, n_samples=100)
    print(f"Local smoothness at X: {L_loc:.4f}")

### Reproducing Paper Results

    # Run all experiments
    python experiments/run_all.py

    # Run individual experiments
    python experiments/adversarial_sampling.py
    python experiments/hard_direction.py
    python experiments/local_smoothness.py

    # Additional experiments
    python experiments/convergence_burnin.py
    python experiments/dimension_scaling.py
    python experiments/cayley_comparison.py

## Key Experiments

### 1. Adversarial vs Random Sampling (Figure 1, Table 2)

Compares empirical Lipschitz constant estimates using random sampling vs. adversarial search:

    python experiments/adversarial_sampling.py --algebras so sl --dims 8 16 --radii 0.5 1.0 1.5 2.0 3.0

Adversarial sampling yields a larger empirical-to-theoretical ratio than random sampling,
with the gain growing with the radius (up to about 3.6x at R = 3).

### 2. Hard Direction Construction (Figure 2, Table 3)

Validates the lower-bound construction from Theorem 5.3 along the optimal hard direction:

    python experiments/hard_direction.py --dim 8 --radii 0.5 1.0 1.5 2.0 3.0 4.0

On sl(8) the empirical exponent 1.8708 matches the theoretical prediction
2 * mu_8 = 2 * sqrt(7/8) to four significant figures, and the empirical-to-theoretical
ratio equals 1.00 at every tested radius.

### 3. Local vs Global Smoothness (Figure 3)

Tracks local smoothness along optimization trajectories:

    python experiments/local_smoothness.py --dim 16 --radius 3.0 --iterations 1000

Local smoothness L_loc is approximately 3 and remains stable, while the global bound
L_global is on the order of 10^5 to 10^6.

## Supported Lie Algebras

| Algebra            | Notation | Type        | Description                       |
|--------------------|----------|-------------|-----------------------------------|
| SpecialOrthogonal  | so(d)    | Compact     | Skew-symmetric matrices           |
| SpecialUnitary     | su(d)    | Compact     | Skew-Hermitian traceless matrices |
| SpecialLinear      | sl(d)    | Non-compact | Traceless matrices                |
| GeneralLinear      | gl(d)    | Non-compact | All d x d matrices                |
| Symplectic         | sp(2k)   | Non-compact | Symplectic matrices               |

## API Reference

### Core Modules

#### algebras.py
- LieAlgebra: Base class for matrix Lie algebras
- SpecialOrthogonal(d): so(d) algebra
- SpecialLinear(d): sl(d) algebra
- project(X, algebra): Orthogonal projection onto algebra

#### exponential.py
- expm(X): Matrix exponential via scaling-and-squaring
- frechet_derivative(X, H): First Frechet derivative Dexp(X)[H]
- frechet_derivative_adjoint(X, Z): Adjoint Dexp(X)^*[Z]
- second_frechet_derivative(X, H1, H2): Second derivative D2exp(X)[H1, H2]

#### smoothness.py
- estimate_global(algebra, radius, n_samples): Global Lipschitz constant
- estimate_local(X, algebra, epsilon): Local Lipschitz constant at X
- theoretical_upper_bound(n, M, R): Upper bound from Theorem 4.1
- theoretical_lower_bound(mu, R): Lower bound from Theorem 5.3

## Citation

If you use this code in your research, please cite:

    @article{kc2026smoothness,
      title={Intrinsic Smoothness Barriers for Optimization via the Matrix Exponential on {L}ie Algebras},
      author={K.C, Sooraj and Mishra, Vivek},
      journal={Indian Journal of Pure and Applied Mathematics},
      year={2026},
      publisher={Springer}
    }

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- This work was supported by Alliance University, Bangalore.

## Contact

- Sooraj K.C - ksoorajPHD23@sam.alliance.edu.in

## Changelog

### v1.0.0 (2026)
- Release accompanying the IJPAM submission.
- Complete experimental reproducibility for all figures and tables.
