# lie-algebra-smoothness
# Intrinsic Smoothness Barriers for Matrix Exponential Optimization on Lie Algebras

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2601.XXXXX)

This repository contains the code to reproduce all experiments from the paper:

> **Intrinsic Smoothness Barriers for Optimization via the Matrix Exponential on Lie Algebras**  
> Sooraj K.C., Vivek Mishra, Sarit Maitra  
> *SIAM Journal on Matrix Analysis and Applications (SIMAX)*, 2026

## Overview

We study the gradient Lipschitz constant (smoothness) for optimization problems of the form:

$$\min_{X \in \mathfrak{g}} L(\exp(X))$$

where $\mathfrak{g}$ is a matrix Lie algebra. Our main results establish:

1. **Upper bounds**: $L_{\text{grad}} \leq C(n,M,R) \cdot e^{2R}$ for general Lie algebras
2. **Lower bounds**: $L_{\text{grad}} \geq c(\mu) \cdot e^{2|\mu|R}$ for non-compact algebras
3. **Compact/Non-compact dichotomy**: Polynomial smoothness for $\mathfrak{so}(d), \mathfrak{su}(d)$ vs. exponential for $\mathfrak{sl}(d), \mathfrak{gl}(d)$

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- JAX (optional, for automatic differentiation)

### Setup

```bash
# Clone the repository
git clone https://github.com/soorajkcphd/lie-algebra-smoothness.git
cd lie-algebra-smoothness

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Repository Structure

```
lie-algebra-smoothness/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── src/
│   └── lie_smoothness/
│       ├── __init__.py
│       ├── algebras.py          # Lie algebra definitions (so, sl, gl, sp, su)
│       ├── exponential.py       # Matrix exponential and Fréchet derivatives
│       ├── objectives.py        # Optimization objectives
│       ├── smoothness.py        # Lipschitz constant estimation
│       ├── optimization.py      # Projected gradient descent
│       └── utils.py             # Helper functions
│
├── experiments/
│   ├── run_all.py               # Run all experiments
│   ├── adversarial_sampling.py  # Figure 1, Table 1
│   ├── hard_direction.py        # Figure 2, Table 2
│   ├── local_smoothness.py      # Figure 3
│   ├── convergence_burnin.py    # Figure SM1
│   ├── dimension_scaling.py     # Table SM1
│   └── cayley_comparison.py     # Figure SM2, Table SM2
│
├── tests/
│   ├── test_algebras.py
│   ├── test_exponential.py
│   └── test_smoothness.py
│
├── figures/                     # Generated figures
│   ├── fig1_adversarial_search.pdf
│   ├── fig2_hard_direction.pdf
│   ├── fig3_local_smoothness.pdf
│   ├── fig4_convergence_burnin.pdf
│   └── fig_cayley_comparison.pdf
│
└── results/                     # Numerical results (CSV)
    └── ...
```

## Quick Start

### Basic Usage

```python
import numpy as np
from lie_smoothness import algebras, exponential, smoothness

# Define a Lie algebra
d = 8
algebra = algebras.SpecialLinear(d)  # sl(d)

# Generate a random element
X = algebra.random_element(radius=2.0)

# Compute matrix exponential and its Fréchet derivative
G = exponential.expm(X)
dG = exponential.frechet_derivative(X, direction=algebra.random_element(radius=0.1))

# Estimate local smoothness
L_loc = smoothness.estimate_local(X, algebra, n_samples=100)
print(f"Local smoothness at X: {L_loc:.4f}")
```

### Reproducing Paper Results

```bash
# Run all experiments (takes ~2-4 hours)
python experiments/run_all.py

# Run individual experiments
python experiments/adversarial_sampling.py  # ~30 min
python experiments/hard_direction.py        # ~10 min
python experiments/local_smoothness.py      # ~20 min

# Supplementary experiments
python experiments/convergence_burnin.py    # 15 min
python experiments/dimension_scaling.py     # ~30 min
python experiments/cayley_comparison.py     # ~15 min
```

## Key Experiments

### 1. Adversarial vs Random Sampling (Figure 1, Table 1)

Compares empirical Lipschitz constant estimates using random sampling vs. adversarial search:

```bash
python experiments/adversarial_sampling.py --algebras so sl --dims 8 16 --radii 0.5 1.0 1.5 2.0 3.0
```

**Key finding**: Adversarial sampling achieves 10×–25× higher empirical Lipschitz constants than random sampling.

### 2. Hard Direction Construction (Figure 2, Table 2)

Validates the lower bound construction from Theorem 5.2:

```bash
python experiments/hard_direction.py --dim 8 --radii 0.5 1.0 1.5 2.0 3.0 4.0
```

**Key finding**: Empirical exponent 1.4142 matches theoretical prediction $2\mu = \sqrt{2}$ to four significant figures.

### 3. Local vs Global Smoothness (Figure 3)

Tracks local smoothness along optimization trajectories:

```bash
python experiments/local_smoothness.py --dim 16 --radius 3.0 --iterations 1000
```

**Key finding**: Local smoothness $L_{\text{loc}} \approx 3$ remains stable while global bound $L_{\text{global}} \sim 10^5$–$10^6$.

## Supported Lie Algebras

| Algebra | Notation | Type | Description |
|---------|----------|------|-------------|
| `SpecialOrthogonal` | $\mathfrak{so}(d)$ | Compact | Skew-symmetric matrices |
| `SpecialUnitary` | $\mathfrak{su}(d)$ | Compact | Skew-Hermitian traceless matrices |
| `SpecialLinear` | $\mathfrak{sl}(d)$ | Non-compact | Traceless matrices |
| `GeneralLinear` | $\mathfrak{gl}(d)$ | Non-compact | All $d \times d$ matrices |
| `Symplectic` | $\mathfrak{sp}(2k)$ | Non-compact | Symplectic matrices |

## API Reference

### Core Modules

#### `algebras.py`
- `LieAlgebra`: Base class for matrix Lie algebras
- `SpecialOrthogonal(d)`: $\mathfrak{so}(d)$ algebra
- `SpecialLinear(d)`: $\mathfrak{sl}(d)$ algebra
- `project(X, algebra)`: Orthogonal projection onto algebra

#### `exponential.py`
- `expm(X)`: Matrix exponential via scaling-and-squaring
- `frechet_derivative(X, H)`: First Fréchet derivative $D\exp(X)[H]$
- `frechet_derivative_adjoint(X, Z)`: Adjoint $D\exp(X)^*[Z]$
- `second_frechet_derivative(X, H1, H2)`: Second derivative $D^2\exp(X)[H_1, H_2]$

#### `smoothness.py`
- `estimate_global(algebra, radius, n_samples)`: Global Lipschitz constant
- `estimate_local(X, algebra, epsilon)`: Local Lipschitz constant at $X$
- `theoretical_upper_bound(n, M, R)`: Upper bound from Theorem 4.1
- `theoretical_lower_bound(mu, R)`: Lower bound from Theorem 5.2

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kc2026smoothness,
  title={Intrinsic Smoothness Barriers for Optimization via the Matrix Exponential on {L}ie Algebras},
  author={K.C, Sooraj and Mishra, Vivek and Maitra, Sarit},
  journal={SIAM Journal on Matrix Analysis and Applications},
  volume={47},
  number={X},
  pages={XXX--XXX},
  year={2026},
  publisher={SIAM}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by Alliance University, Bangalore
- We thank the reviewers for their constructive feedback

## Contact

- **Sooraj K.C.** - sooraj.kc@alliance.edu.in
- **Vivek Mishra** - vivek.mishra@alliance.edu.in  
- **Sarit Maitra** - sarit.maitra@alliance.edu.in

## Changelog

### v1.0.0 (January 2026)
- Initial release accompanying SIMAX publication
- Complete experimental reproducibility for all figures and tables
