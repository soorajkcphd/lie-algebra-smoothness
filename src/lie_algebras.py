"""
Core utilities for Lie algebra operations and matrix exponential optimization.

This module provides:
- Lie algebra projections (so(d), sl(d), gl(d))
- Matrix exponential and its Fréchet derivatives
- Gradient computation for the optimization objective
- Smoothness estimation utilities
"""

import numpy as np
from scipy.linalg import expm, norm
from typing import Tuple, Optional, Callable
import warnings


# =============================================================================
# Lie Algebra Projections
# =============================================================================

def proj_so(A: np.ndarray) -> np.ndarray:
    """Project matrix A onto so(d) (skew-symmetric matrices)."""
    return 0.5 * (A - A.T)


def proj_sl(A: np.ndarray) -> np.ndarray:
    """Project matrix A onto sl(d) (traceless matrices)."""
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d)


def proj_gl(A: np.ndarray) -> np.ndarray:
    """Project matrix A onto gl(d) (identity projection)."""
    return A.copy()


def proj_ball(X: np.ndarray, R: float) -> np.ndarray:
    """Project X onto the Frobenius ball of radius R."""
    norm_X = norm(X, 'fro')
    if norm_X <= R:
        return X
    return X * (R / norm_X)


def get_projection(algebra: str) -> Callable:
    """Get the projection function for a given Lie algebra."""
    projections = {
        'so': proj_so,
        'sl': proj_sl,
        'gl': proj_gl,
    }
    if algebra not in projections:
        raise ValueError(f"Unknown algebra: {algebra}. Choose from {list(projections.keys())}")
    return projections[algebra]


# =============================================================================
# Matrix Exponential and Fréchet Derivatives
# =============================================================================

def matrix_exp(X: np.ndarray) -> np.ndarray:
    """Compute matrix exponential exp(X)."""
    return expm(X)


def frechet_exp(X: np.ndarray, H: np.ndarray, num_points: int = 50) -> np.ndarray:
    """
    Compute the Fréchet derivative Dexp(X)[H] using numerical integration.
    
    Uses the integral formula:
        Dexp(X)[H] = ∫₀¹ exp((1-s)X) H exp(sX) ds
    """
    d = X.shape[0]
    result = np.zeros((d, d), dtype=complex if np.iscomplexobj(X) else float)
    
    # Simpson's rule integration
    s_vals = np.linspace(0, 1, num_points)
    ds = 1.0 / (num_points - 1)
    
    for i, s in enumerate(s_vals):
        integrand = expm((1 - s) * X) @ H @ expm(s * X)
        
        # Simpson's rule weights
        if i == 0 or i == num_points - 1:
            weight = 1.0
        elif i % 2 == 1:
            weight = 4.0
        else:
            weight = 2.0
        
        result += weight * integrand
    
    result *= ds / 3.0
    return np.real(result) if not np.iscomplexobj(X) else result


def frechet_exp_adjoint(X: np.ndarray, Z: np.ndarray, num_points: int = 50) -> np.ndarray:
    """
    Compute the adjoint Dexp(X)*[Z] using numerical integration.
    
    Uses the formula:
        Dexp(X)*[Z] = ∫₀¹ exp(sX^T) Z exp((1-s)X^T) ds
    """
    return frechet_exp(X.T, Z, num_points)


def hessian_exp_direction(X: np.ndarray, H: np.ndarray, num_points: int = 30) -> np.ndarray:
    """
    Compute D²exp(X)[H,H] (second Fréchet derivative in direction H).
    
    Uses double integral formula.
    """
    d = X.shape[0]
    result = np.zeros((d, d), dtype=complex if np.iscomplexobj(X) else float)
    
    s_vals = np.linspace(0, 1, num_points)
    ds = 1.0 / (num_points - 1)
    
    for i, s in enumerate(s_vals):
        for j, u in enumerate(s_vals[:i+1]):
            if u > s:
                continue
            
            # Two symmetric terms
            term1 = expm((1-s)*X) @ H @ expm((s-u)*X) @ H @ expm(u*X)
            term2 = expm((1-s)*X) @ H @ expm((s-u)*X) @ H @ expm(u*X)
            
            # Weights for 2D Simpson's rule (simplified)
            weight = ds * ds
            if i == 0 or i == num_points - 1:
                weight *= 0.5
            if j == 0 or j == i:
                weight *= 0.5
            
            result += weight * (term1 + term2)
    
    return np.real(result) if not np.iscomplexobj(X) else result


# =============================================================================
# Optimization Objective and Gradient
# =============================================================================

def objective(X: np.ndarray, T: np.ndarray, V: np.ndarray, 
              lambd: float = 0.0) -> float:
    """
    Compute the objective function:
        L(X) = (1/2) Σⱼ ||exp(X)vⱼ - Tvⱼ||² + (λ/2)||X||²
    
    Args:
        X: Current point in Lie algebra (d x d)
        T: Target transformation (d x d)
        V: Probe vectors as columns (d x n)
        lambd: Regularization parameter
    
    Returns:
        Objective value
    """
    expX = expm(X)
    residual = expX @ V - T @ V
    loss = 0.5 * np.sum(residual ** 2)
    reg = 0.5 * lambd * norm(X, 'fro') ** 2
    return loss + reg


def gradient(X: np.ndarray, T: np.ndarray, V: np.ndarray,
             algebra: str = 'gl', lambd: float = 0.0,
             num_points: int = 50) -> np.ndarray:
    """
    Compute the gradient of the objective.
    
    Args:
        X: Current point in Lie algebra (d x d)
        T: Target transformation (d x d)
        V: Probe vectors as columns (d x n)
        algebra: Lie algebra type ('so', 'sl', 'gl')
        lambd: Regularization parameter
        num_points: Integration points for Fréchet derivative
    
    Returns:
        Gradient ∇L(X) projected onto the Lie algebra
    """
    expX = expm(X)
    
    # Gradient of outer objective: ∇f(G) = Σⱼ (Gvⱼ - Tvⱼ)vⱼᵀ
    residual = expX @ V - T @ V  # d x n
    grad_f = residual @ V.T  # d x d
    
    # Chain rule: Dexp(X)*[∇f(exp(X))]
    grad_unproj = frechet_exp_adjoint(X, grad_f, num_points)
    
    # Project onto Lie algebra
    proj = get_projection(algebra)
    grad_L = proj(grad_unproj) + lambd * X
    
    return grad_L


def gradient_finite_diff(X: np.ndarray, T: np.ndarray, V: np.ndarray,
                         algebra: str = 'gl', lambd: float = 0.0,
                         eps: float = 1e-6) -> np.ndarray:
    """
    Compute gradient using finite differences (for verification).
    """
    d = X.shape[0]
    grad = np.zeros_like(X)
    proj = get_projection(algebra)
    
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d))
            E[i, j] = 1.0
            E = proj(E)
            if norm(E, 'fro') < 1e-10:
                continue
            E = E / norm(E, 'fro')
            
            f_plus = objective(X + eps * E, T, V, lambd)
            f_minus = objective(X - eps * E, T, V, lambd)
            grad += ((f_plus - f_minus) / (2 * eps)) * E
    
    return proj(grad)


# =============================================================================
# Smoothness Estimation
# =============================================================================

def estimate_local_lipschitz(X: np.ndarray, T: np.ndarray, V: np.ndarray,
                              algebra: str = 'gl', lambd: float = 0.0,
                              num_directions: int = 20, delta: float = 0.01) -> float:
    """
    Estimate local Lipschitz constant of gradient at X via random probing.
    
    L_loc(X) ≈ max_k ||∇L(X + δUₖ) - ∇L(X)|| / δ
    """
    d = X.shape[0]
    proj = get_projection(algebra)
    
    grad_X = gradient(X, T, V, algebra, lambd)
    
    max_ratio = 0.0
    for _ in range(num_directions):
        # Random direction in Lie algebra
        U = np.random.randn(d, d)
        U = proj(U)
        U = U / norm(U, 'fro')
        
        # Perturbed gradient
        grad_perturbed = gradient(X + delta * U, T, V, algebra, lambd)
        
        ratio = norm(grad_perturbed - grad_X, 'fro') / delta
        max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def estimate_global_lipschitz(T: np.ndarray, V: np.ndarray, R: float,
                               algebra: str = 'gl', lambd: float = 0.0,
                               num_pairs: int = 100) -> float:
    """
    Estimate global Lipschitz constant via sampling pairs.
    
    L_grad ≈ max_{X,Y} ||∇L(X) - ∇L(Y)|| / ||X - Y||
    """
    d = T.shape[0]
    proj = get_projection(algebra)
    
    max_ratio = 0.0
    for _ in range(num_pairs):
        # Sample two points in ball
        X = np.random.randn(d, d)
        X = proj(X)
        X = X * (R * np.random.rand() / max(norm(X, 'fro'), 1e-10))
        
        Y = np.random.randn(d, d)
        Y = proj(Y)
        Y = Y * (R * np.random.rand() / max(norm(Y, 'fro'), 1e-10))
        
        diff_XY = norm(X - Y, 'fro')
        if diff_XY < 1e-10:
            continue
        
        grad_X = gradient(X, T, V, algebra, lambd)
        grad_Y = gradient(Y, T, V, algebra, lambd)
        
        ratio = norm(grad_X - grad_Y, 'fro') / diff_XY
        max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def theoretical_lipschitz_bound(R: float, M: float, n: int, 
                                 lambd: float = 0.0,
                                 algebra: str = 'gl') -> float:
    """
    Compute theoretical upper bound on Lipschitz constant.
    
    L_grad ≤ 2n * e^{2R} * (2M + 2R + 1) + 2λ
    """
    if algebra == 'so':
        # Compact case: independent of R
        d = int(np.sqrt(n))  # Approximate
        return 2 * n * (M + np.sqrt(d)) ** 2 + 2 * lambd
    else:
        # Non-compact case
        return 2 * n * np.exp(2 * R) * (2 * M + 2 * R + 1) + 2 * lambd


# =============================================================================
# Hard Direction Construction (for lower bound)
# =============================================================================

def hard_direction_sl(d: int) -> Tuple[np.ndarray, float]:
    """
    Construct the hard direction X₀ = diag(1,-1,0,...,0)/√2 for sl(d).
    
    Returns:
        X0: The hard direction (unit Frobenius norm, traceless)
        mu: The real eigenvalue (1/√2)
    """
    X0 = np.zeros((d, d))
    X0[0, 0] = 1.0
    X0[1, 1] = -1.0
    X0 = X0 / np.sqrt(2)  # Normalize to unit Frobenius norm
    
    mu = 1.0 / np.sqrt(2)
    return X0, mu


def hard_direction_gl(d: int) -> Tuple[np.ndarray, float]:
    """
    Construct hard direction for gl(d): X₀ = e₁e₁ᵀ.
    
    Returns:
        X0: The hard direction
        mu: The eigenvalue (1.0)
    """
    X0 = np.zeros((d, d))
    X0[0, 0] = 1.0
    mu = 1.0
    return X0, mu


def hessian_along_hard_direction(t: float, X0: np.ndarray, 
                                  mu: float, v: np.ndarray) -> float:
    """
    Compute d²L/dt²(tX₀) for the lower bound construction.
    
    For L(X) = (1/2)||exp(X)v - w||² with w = exp(R*X₀)v:
        d²L/dt²(tX₀)|_{t=R} = μ² * e^{2μR}
    """
    return mu ** 2 * np.exp(2 * mu * t)


# =============================================================================
# Cayley Transform (for comparison)
# =============================================================================

def cayley(X: np.ndarray) -> np.ndarray:
    """
    Compute Cayley transform: Cay(X) = (I + X/2)(I - X/2)^{-1}
    
    Valid for ||X||_2 < 2.
    """
    d = X.shape[0]
    I = np.eye(d)
    return np.linalg.solve(I - X/2, I + X/2)


def cayley_derivative(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute Cayley derivative: DCay(X)[H] = (I - X/2)^{-1} H (I - X/2)^{-1}
    """
    d = X.shape[0]
    I = np.eye(d)
    inv = np.linalg.inv(I - X/2)
    return inv @ H @ inv


if __name__ == "__main__":
    # Quick test
    d = 4
    X = proj_so(np.random.randn(d, d))
    H = proj_so(np.random.randn(d, d))
    
    print(f"X is skew-symmetric: {np.allclose(X, -X.T)}")
    print(f"exp(X) is orthogonal: {np.allclose(expm(X) @ expm(X).T, np.eye(d))}")
    print(f"||Dexp(X)[H]||_F = {norm(frechet_exp(X, H), 'fro'):.4f}")
    print(f"||H||_F = {norm(H, 'fro'):.4f}")
