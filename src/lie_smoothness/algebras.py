#
# Lie Algebra Definitions

# Matrix Lie algebras studied in the paper:
# - so(d): Skew-symmetric matrices (compact)
# - su(d): Skew-Hermitian traceless matrices (compact)  
# - sl(d): Traceless matrices (non-compact)
# - gl(d): All matrices (non-compact)
# - sp(2k): Symplectic matrices (non-compact)
#

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class LieAlgebra(ABC):
    #Abstract base class for matrix Lie algebras.#
    
    def __init__(self, d: int):
        #
        # Parameters
        # ----------
        # d : int
        #     Matrix dimension (d x d matrices)
        #
        self.d = d
    
    @property
    @abstractmethod
    def name(self) -> str:
        #Algebra name (e.g., 'so(8)').#
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        #Dimension as a vector space.#
        pass
    
    @property
    @abstractmethod
    def is_compact(self) -> bool:
        #Whether the corresponding Lie group is compact.#
        pass
    
    @abstractmethod
    def project(self, X: np.ndarray) -> np.ndarray:
        #Orthogonal projection onto the algebra.#
        pass
    
    def random_element(self, radius: float = 1.0, 
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        #
        # Generate random element with given Frobenius norm.
        # Parameters
        # ----------
        # radius : float
        #     Desired Frobenius norm
        # rng : Generator, optional
        #     NumPy random generator for reproducibility
        #
        if rng is None:
            rng = np.random.default_rng()
        X = rng.standard_normal((self.d, self.d))
        X = self.project(X)
        norm = np.linalg.norm(X, 'fro')
        if norm > 1e-10:
            X = X * (radius / norm)
        return X
    
    def contains(self, X: np.ndarray, tol: float = 1e-10) -> bool:
        #Check if matrix is in the algebra.#
        return np.linalg.norm(X - self.project(X), 'fro') < tol
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.d})"


class SpecialOrthogonal(LieAlgebra):
    #
    # so(d): Skew-symmetric matrices.
    
    # X ∈ so(d) ⟺ X^T = -X
    
    # This is a COMPACT Lie algebra. The matrix exponential maps to SO(d).
    #
    
    @property
    def name(self) -> str:
        return f"so({self.d})"
    
    @property
    def dim(self) -> int:
        return self.d * (self.d - 1) // 2
    
    @property
    def is_compact(self) -> bool:
        return True
    
    def project(self, X: np.ndarray) -> np.ndarray:
        #Project to skew-symmetric: (X - X^T) / 2#
        return (X - X.T) / 2


class SpecialUnitary(LieAlgebra):
    
    # su(d): Skew-Hermitian traceless matrices.
    # X ∈ su(d) ⟺ X^H = -X and tr(X) = 0  
    # This is a COMPACT Lie algebra. The matrix exponential maps to SU(d).
    
    @property
    def name(self) -> str:
        return f"su({self.d})"
    
    @property
    def dim(self) -> int:
        return self.d * self.d - 1
    
    @property
    def is_compact(self) -> bool:
        return True
    
    def project(self, X: np.ndarray) -> np.ndarray:
        #Project to skew-Hermitian traceless.#
        X = (X - X.conj().T) / 2  # Skew-Hermitian
        X = X - np.trace(X) / self.d * np.eye(self.d)  # Traceless
        return X
    
    def random_element(self, radius: float = 1.0,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        X = rng.standard_normal((self.d, self.d)) + 1j * rng.standard_normal((self.d, self.d))
        X = self.project(X)
        norm = np.linalg.norm(X, 'fro')
        if norm > 1e-10:
            X = X * (radius / norm)
        return X


class SpecialLinear(LieAlgebra):

    # sl(d): Traceless matrices.
    # X ∈ sl(d) ⟺ tr(X) = 0
    # This is a NON-COMPACT Lie algebra. The matrix exponential maps to SL(d).
    
    @property
    def name(self) -> str:
        return f"sl({self.d})"
    
    @property
    def dim(self) -> int:
        return self.d * self.d - 1
    
    @property
    def is_compact(self) -> bool:
        return False
    
    def project(self, X: np.ndarray) -> np.ndarray:
        #Project to traceless: X - tr(X)/d * I#
        return X - np.trace(X) / self.d * np.eye(self.d)
    
    def hard_direction(self) -> np.ndarray:
    
        # Hard direction achieving maximal eigenvalue |μ| = 1/√2.
        # X_0 = diag(1, -1, 0, ..., 0) / √2
        # Used in lower bound construction (Theorem 5.2).
        
        X = np.zeros((self.d, self.d))
        X[0, 0] = 1.0
        X[1, 1] = -1.0
        return X / np.sqrt(2)
    
    def max_real_eigenvalue(self) -> float:
        #Maximum real eigenvalue for unit-norm traceless matrices: 1/√2.#
        return 1.0 / np.sqrt(2)


class GeneralLinear(LieAlgebra):
    
    # gl(d): All d × d matrices.
    # This is a NON-COMPACT Lie algebra. The matrix exponential maps to GL(d).
    
    
    @property
    def name(self) -> str:
        return f"gl({self.d})"
    
    @property
    def dim(self) -> int:
        return self.d * self.d
    
    @property
    def is_compact(self) -> bool:
        return False
    
    def project(self, X: np.ndarray) -> np.ndarray:
        #Identity (all matrices in gl(d)).#
        return X.copy()


class Symplectic(LieAlgebra):
    
    # sp(2k): Symplectic matrices satisfying X^T J + J X = 0.
    # J = [[0, I_k], [-I_k, 0]]
    # This is a NON-COMPACT Lie algebra.
    # Parameters
    # ----------
    # k : int
    #     Half-dimension (matrices are 2k × 2k)
    
    
    def __init__(self, k: int):
        self.k = k
        super().__init__(d=2*k)
        I = np.eye(k)
        Z = np.zeros((k, k))
        self.J = np.block([[Z, I], [-I, Z]])
    
    @property
    def name(self) -> str:
        return f"sp({2*self.k})"
    
    @property
    def dim(self) -> int:
        return self.k * (2 * self.k + 1)
    
    @property
    def is_compact(self) -> bool:
        return False
    
    def project(self, X: np.ndarray) -> np.ndarray:
        #Project to sp(2k): X^T J + J X = 0.#
        # sp(2k) = {[[A, B], [C, -A^T]] : B = B^T, C = C^T}
        k = self.k
        A = X[:k, :k]
        B = X[:k, k:]
        C = X[k:, :k]
        D = X[k:, k:]
        
        B_sym = (B + B.T) / 2
        C_sym = (C + C.T) / 2
        A_new = (A - D.T) / 2
        
        return np.block([[A_new, B_sym], [C_sym, -A_new.T]])


def get_algebra(name: str, d: int) -> LieAlgebra:
    #
    # Factory function to get algebra by name.
    # Parameters
    # ----------
    # name : str
    #     Algebra name: 'so', 'su', 'sl', 'gl', or 'sp'
    # d : int
    #     Dimension (for sp, this is 2k)
    #
    algebras = {
        'so': SpecialOrthogonal,
        'su': SpecialUnitary,
        'sl': SpecialLinear,
        'gl': GeneralLinear,
    }
    
    if name == 'sp':
        if d % 2 != 0:
            raise ValueError("sp(2k) requires even dimension")
        return Symplectic(d // 2)
    
    if name not in algebras:
        raise ValueError(f"Unknown algebra: {name}. Choose from {list(algebras.keys())}")
    
    return algebras[name](d)
