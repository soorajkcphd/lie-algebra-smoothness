
# Matrix Exponential and Fréchet Derivatives

# Implements:
# - Matrix exponential via SciPy
# - First Fréchet derivative: D exp(X)[H] = ∫₀¹ exp((1-s)X) H exp(sX) ds
# - Adjoint: D exp(X)*[Z] = ∫₀¹ exp(sX)^T Z exp((1-s)X)^T ds  
# - Second Fréchet derivative: D² exp(X)[H, K]

# References:
# - Higham, "Functions of Matrices", SIAM 2008
# - Al-Mohy & Higham, SIMAX 2009, 2010
# - Najfeld & Havel, Adv. Appl. Math. 1995


import numpy as np
from scipy.linalg import expm as scipy_expm
from typing import Optional, Tuple


def expm(X: np.ndarray) -> np.ndarray:
    return scipy_expm(X)


def frechet_derivative(X: np.ndarray, H: np.ndarray, 
                       n_quad: int = 50) -> np.ndarray:
    
    # First Fréchet derivative D exp(X)[H].
    
    # Uses the integral formula:
    #     D exp(X)[H] = ∫₀¹ exp((1-s)X) H exp(sX) ds
    
    # Parameters
    # ----------
    # X : np.ndarray
    #     Evaluation point (d × d)
    # H : np.ndarray
    #     Direction (d × d)
    # n_quad : int
    #     Gauss-Legendre quadrature points
        
    # Returns
    # -------
    # np.ndarray
    #     D exp(X)[H]
    
    d = X.shape[0]
    is_complex = np.iscomplexobj(X) or np.iscomplexobj(H)
    result = np.zeros((d, d), dtype=complex if is_complex else float)
    
    # Gauss-Legendre quadrature on [0, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = (nodes + 1) / 2  # Map [-1, 1] → [0, 1]
    weights = weights / 2
    
    for s, w in zip(nodes, weights):
        exp_1_s = scipy_expm((1 - s) * X)
        exp_s = scipy_expm(s * X)
        result += w * (exp_1_s @ H @ exp_s)
    
    return result.real if not is_complex else result


def frechet_derivative_adjoint(X: np.ndarray, Z: np.ndarray,
                                n_quad: int = 50) -> np.ndarray:

    # Adjoint of Fréchet derivative D exp(X)*[Z].
    
    # Satisfies: ⟨D exp(X)[H], Z⟩_F = ⟨H, D exp(X)*[Z]⟩_F
    
    # Uses the integral formula:
    #     D exp(X)*[Z] = ∫₀¹ exp(sX)^T Z exp((1-s)X)^T ds
    
    # Parameters
    # ----------
    # X : np.ndarray
    #     Evaluation point (d × d)
    # Z : np.ndarray
    #     Input to adjoint (d × d)
    # n_quad : int
    #     Gauss-Legendre quadrature points
        
    # Returns
    # -------
    # np.ndarray
    #     D exp(X)*[Z]
    
    d = X.shape[0]
    is_complex = np.iscomplexobj(X) or np.iscomplexobj(Z)
    result = np.zeros((d, d), dtype=complex if is_complex else float)
    
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    
    for s, w in zip(nodes, weights):
        exp_s = scipy_expm(s * X)
        exp_1_s = scipy_expm((1 - s) * X)
        result += w * (exp_s.T @ Z @ exp_1_s.T)
    
    return result.real if not is_complex else result


def second_frechet_derivative(X: np.ndarray, H: np.ndarray, K: np.ndarray,
                              n_quad: int = 30) -> np.ndarray:

    # Second Fréchet derivative D² exp(X)[H, K].
    
    # Uses double integral formula.
    
    # Parameters
    # ----------
    # X : np.ndarray
    #     Evaluation point
    # H, K : np.ndarray
    #     Directions
    # n_quad : int
    #     Quadrature points per dimension
        
    # Returns
    # -------
    # np.ndarray
    #     D² exp(X)[H, K]
    d = X.shape[0]
    is_complex = any(np.iscomplexobj(arr) for arr in [X, H, K])
    result = np.zeros((d, d), dtype=complex if is_complex else float)
    
    s_nodes, s_weights = np.polynomial.legendre.leggauss(n_quad)
    s_nodes = (s_nodes + 1) / 2
    s_weights = s_weights / 2
    
    for s, ws in zip(s_nodes, s_weights):
        if s < 1e-10:
            continue
            
        u_nodes, u_weights = np.polynomial.legendre.leggauss(n_quad)
        u_nodes = s * (u_nodes + 1) / 2  # Map to [0, s]
        u_weights = s * u_weights / 2
        
        exp_1_s = scipy_expm((1 - s) * X)
        
        for u, wu in zip(u_nodes, u_weights):
            exp_s_u = scipy_expm((s - u) * X)
            exp_u = scipy_expm(u * X)
            
            # Symmetric sum
            term1 = exp_1_s @ H @ exp_s_u @ K @ exp_u
            term2 = exp_1_s @ K @ exp_s_u @ H @ exp_u
            result += ws * wu * (term1 + term2)
    
    return result.real if not is_complex else result


def operator_norm_estimate(X: np.ndarray, n_iter: int = 50) -> float:
    # Estimate ||D exp(X)||_op via power iteration.
    # Parameters
    # ----------
    # X : np.ndarray
    #     Evaluation point
    # n_iter : int
    #     Number of iterations
    # Returns
    # -------
    # float
    #     Estimated operator norm
    d = X.shape[0]
    H = np.random.randn(d, d)
    H /= np.linalg.norm(H, 'fro')
    
    for _ in range(n_iter):
        DH = frechet_derivative(X, H)
        H_new = frechet_derivative_adjoint(X, DH)
        norm = np.linalg.norm(H_new, 'fro')
        if norm > 1e-10:
            H = H_new / norm
    
    return np.linalg.norm(frechet_derivative(X, H), 'fro')


def verify_adjoint(X: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float]:
    
    #Verify adjoint: ⟨D exp(X)[H], Z⟩ = ⟨H, D exp(X)*[Z]⟩
    #Returns (passed, relative_error)
    d = X.shape[0]
    H = np.random.randn(d, d)
    Z = np.random.randn(d, d)
    
    left = np.sum(frechet_derivative(X, H) * Z)
    right = np.sum(H * frechet_derivative_adjoint(X, Z))
    
    rel_err = abs(left - right) / max(abs(left), abs(right), 1e-10)
    return rel_err < tol, rel_err


def theoretical_dexp_bound(R: float) -> float:
    
    #Theoretical bound ||D exp(X)||_op ≤ exp(R) for ||X||_F ≤ R.
    #This is the key factor leading to O(exp(2R)) smoothness.
    
    return np.exp(R)
