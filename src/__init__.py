"""
Lie Algebra Smoothness Analysis Package

This package provides tools for analyzing smoothness barriers in optimization
over matrix Lie algebras via the exponential map.

Main modules:
- lie_algebras: Core utilities for Lie algebra operations and derivatives
"""

from .lie_algebras import (
    proj_so, proj_sl, proj_gl, proj_ball,
    matrix_exp, frechet_exp, frechet_exp_adjoint,
    objective, gradient,
    estimate_local_lipschitz, estimate_global_lipschitz,
    theoretical_lipschitz_bound,
    hard_direction_sl, hard_direction_gl,
    cayley, cayley_derivative
)

__version__ = "1.0.0"
__author__ = "Sooraj K.C., Vivek Mishra, Sarit Maitra"
