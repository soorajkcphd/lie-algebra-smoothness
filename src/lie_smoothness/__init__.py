"""
Lie Algebra Smoothness Analysis Package

Analyze smoothness constants for optimization via the matrix exponential on Lie algebras.

Main modules:
- algebras: Lie algebra definitions (so, sl, gl, sp, su)
- exponential: Matrix exponential and Fréchet derivatives
- objectives: Optimization objective functions
- smoothness: Lipschitz constant estimation
- optimization: Projected gradient descent
"""

from .algebras import (
    LieAlgebra,
    SpecialOrthogonal,
    SpecialUnitary,
    SpecialLinear,
    GeneralLinear,
    Symplectic,
)
from .exponential import (
    expm,
    frechet_derivative,
    frechet_derivative_adjoint,
    second_frechet_derivative,
)
from .smoothness import (
    estimate_global,
    estimate_local,
    adversarial_search,
    theoretical_upper_bound,
    theoretical_lower_bound,
)

__version__ = "1.0.0"
__author__ = "Sooraj K.C., Vivek Mishra, Sarit Maitra"

__all__ = [
    # Algebras
    "LieAlgebra",
    "SpecialOrthogonal",
    "SpecialUnitary",
    "SpecialLinear",
    "GeneralLinear",
    "Symplectic",
    # Exponential
    "expm",
    "frechet_derivative",
    "frechet_derivative_adjoint",
    "second_frechet_derivative",
    # Smoothness
    "estimate_global",
    "estimate_local",
    "adversarial_search",
    "theoretical_upper_bound",
    "theoretical_lower_bound",
]
