
# Smoothness (Lipschitz Constant) Estimation

# Estimate global and local gradient Lipschitz constants for:
#     L(X) = (1/2) Σⱼ ||exp(X) vⱼ - T vⱼ||²

# Methods:
# - Random sampling
# - Adversarial search (gradient ascent on Lipschitz ratio)
# - Theoretical bounds (Theorems 4.1, 4.2, 5.2)


import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm

from .algebras import LieAlgebra
from .exponential import expm, frechet_derivative_adjoint


def fitting_objective(X: np.ndarray, v: np.ndarray, T: np.ndarray) -> float:

    expX = expm(X)
    residual = expX @ v - T @ v
    return 0.5 * np.sum(residual ** 2)


def fitting_gradient(X: np.ndarray, v: np.ndarray, T: np.ndarray,
                     algebra: LieAlgebra) -> np.ndarray:

    expX = expm(X)
    residual = expX @ v - T @ v  # d × n
    F = residual @ v.T  # d × d
    grad = frechet_derivative_adjoint(X, F)
    return algebra.project(grad)


def lipschitz_ratio(X: np.ndarray, Y: np.ndarray, v: np.ndarray, T: np.ndarray,
                    algebra: LieAlgebra) -> float:

    grad_X = fitting_gradient(X, v, T, algebra)
    grad_Y = fitting_gradient(Y, v, T, algebra)
    
    grad_diff = np.linalg.norm(grad_X - grad_Y, 'fro')
    point_diff = np.linalg.norm(X - Y, 'fro')
    
    if point_diff < 1e-12:
        return 0.0
    return grad_diff / point_diff


def estimate_global(algebra: LieAlgebra, v: np.ndarray, T: np.ndarray,
                    radius: float, n_samples: int = 1000,
                    method: str = 'random',
                    rng: Optional[np.random.Generator] = None) -> float:

    if rng is None:
        rng = np.random.default_rng()
    
    max_ratio = 0.0
    
    if method == 'random':
        for _ in range(n_samples):
            X = algebra.random_element(radius=rng.uniform(0, radius), rng=rng)
            Y = algebra.random_element(radius=rng.uniform(0, radius), rng=rng)
            ratio = lipschitz_ratio(X, Y, v, T, algebra)
            max_ratio = max(max_ratio, ratio)
    
    elif method == 'adversarial':
        max_ratio = adversarial_search(algebra, v, T, radius, 
                                        n_restarts=10, n_steps=100, rng=rng)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return max_ratio


def adversarial_search(algebra: LieAlgebra, v: np.ndarray, T: np.ndarray,
                       radius: float, n_restarts: int = 10, n_steps: int = 100,
                       step_size: float = 0.1,
                       rng: Optional[np.random.Generator] = None) -> float:

    if rng is None:
        rng = np.random.default_rng()
    
    max_ratio = 0.0
    
    for _ in range(n_restarts):
        # Initialize
        X = algebra.random_element(radius=radius * 0.9, rng=rng)
        Y = algebra.random_element(radius=radius * 0.9, rng=rng)
        
        for step in range(n_steps):
            # Compute current ratio
            grad_X = fitting_gradient(X, v, T, algebra)
            grad_Y = fitting_gradient(Y, v, T, algebra)
            
            diff = X - Y
            grad_diff = grad_X - grad_Y
            
            diff_norm = np.linalg.norm(diff, 'fro')
            if diff_norm < 1e-10:
                # Reinitialize if points collapsed
                Y = algebra.random_element(radius=radius * 0.9, rng=rng)
                continue
            
            ratio = np.linalg.norm(grad_diff, 'fro') / diff_norm
            max_ratio = max(max_ratio, ratio)
            
            # Gradient ascent step on ratio
            # Move X and Y to increase ||grad_X - grad_Y|| / ||X - Y||
            perturbation = algebra.random_element(radius=step_size, rng=rng)
            
            # Try moving X
            X_new = X + perturbation
            if np.linalg.norm(X_new, 'fro') <= radius:
                new_ratio = lipschitz_ratio(X_new, Y, v, T, algebra)
                if new_ratio > ratio:
                    X = X_new
            
            # Try moving Y
            Y_new = Y - perturbation
            if np.linalg.norm(Y_new, 'fro') <= radius:
                new_ratio = lipschitz_ratio(X, Y_new, v, T, algebra)
                if new_ratio > ratio:
                    Y = Y_new
    
    return max_ratio


def estimate_local(X: np.ndarray, algebra: LieAlgebra, v: np.ndarray, 
                   T: np.ndarray, epsilon: float = 0.01,
                   n_directions: int = 50,
                   rng: Optional[np.random.Generator] = None) -> float:

    if rng is None:
        rng = np.random.default_rng()
    
    max_ratio = 0.0
    
    for _ in range(n_directions):
        direction = algebra.random_element(radius=1.0, rng=rng)
        Y = X + epsilon * direction
        ratio = lipschitz_ratio(X, Y, v, T, algebra)
        max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def theoretical_upper_bound(n: int, M: float, R: float) -> float:
    C = 2 * n * (2 * M + 2 * R + 1)
    return C * np.exp(2 * R)


def theoretical_lower_bound(mu: float, R: float) -> float:

    c_mu = mu ** 2 / 16
    return c_mu * np.exp(2 * abs(mu) * R)


def theoretical_compact_bound(n: int, M: float, d: int) -> float:

    return n * (M + np.sqrt(d)) ** 2


def fit_exponential_scaling(radii: List[float], 
                            lipschitz_values: List[float]) -> Tuple[float, float]:
    log_L = np.log(lipschitz_values)
    R = np.array(radii)
    
    # Linear fit: log(L) = log(c) + β * R
    A = np.vstack([np.ones_like(R), R]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_L, rcond=None)
    
    c = np.exp(coeffs[0])
    beta = coeffs[1]
    
    return c, beta
