
#Tests for Lie Algebra Definitions

import numpy as np
import pytest
from lie_smoothness.algebras import (
    SpecialOrthogonal, SpecialUnitary, SpecialLinear, 
    GeneralLinear, Symplectic, get_algebra
)


class TestSpecialOrthogonal:
    #Tests for so(d) algebra."
    
    def test_projection_skew_symmetric(self):
        #Projected matrices should be skew-symmetric."
        d = 5
        algebra = SpecialOrthogonal(d)
        X = np.random.randn(d, d)
        P = algebra.project(X)
        
        assert np.allclose(P, -P.T), "Projection not skew-symmetric"
    
    def test_projection_idempotent(self):
        #Projection should be idempotent."
        d = 5
        algebra = SpecialOrthogonal(d)
        X = np.random.randn(d, d)
        
        P1 = algebra.project(X)
        P2 = algebra.project(P1)
        
        assert np.allclose(P1, P2), "Projection not idempotent"
    
    def test_random_element_norm(self):
        #Random element should have specified norm."
        d = 5
        algebra = SpecialOrthogonal(d)
        radius = 2.5
        
        X = algebra.random_element(radius=radius)
        
        assert np.isclose(np.linalg.norm(X, 'fro'), radius, rtol=1e-10)
    
    def test_random_element_in_algebra(self):
        #Random element should be in the algebra."
        d = 5
        algebra = SpecialOrthogonal(d)
        X = algebra.random_element()
        
        assert algebra.contains(X)
    
    def test_dimension(self):
        #Check dimension formula: dim(so(d)) = d(d-1)/2.
        for d in [3, 5, 8, 10]:
            algebra = SpecialOrthogonal(d)
            assert algebra.dim == d * (d - 1) // 2
    
    def test_is_compact(self):
        #so(d) should be compact."
        algebra = SpecialOrthogonal(5)
        assert algebra.is_compact


class TestSpecialLinear:
    #Tests for sl(d) algebra."
    
    def test_projection_traceless(self):
        #Projected matrices should be traceless.
        d = 5
        algebra = SpecialLinear(d)
        X = np.random.randn(d, d)
        P = algebra.project(X)
        
        assert np.isclose(np.trace(P), 0, atol=1e-12), "Projection not traceless"
    
    def test_hard_direction_properties(self):
        #Hard direction should have correct properties
        d = 8
        algebra = SpecialLinear(d)
        X0 = algebra.hard_direction()
        
        # Unit norm
        assert np.isclose(np.linalg.norm(X0, 'fro'), 1.0)
        
        # Traceless
        assert np.isclose(np.trace(X0), 0, atol=1e-12)
        
        # Eigenvalue |μ| = 1/√2
        eigenvalues = np.linalg.eigvals(X0)
        max_real_eig = max(abs(np.real(eigenvalues)))
        assert np.isclose(max_real_eig, 1/np.sqrt(2), rtol=1e-10)
    
    def test_is_not_compact(self):
        #sl(d) should not be compact."
        algebra = SpecialLinear(5)
        assert not algebra.is_compact


class TestSymplectic:
    #Tests for sp(2k) algebra."
    
    def test_projection_symplectic_condition(self):
        #Projected matrices should satisfy X^T J + J X = 0."
        k = 3
        algebra = Symplectic(k)
        X = np.random.randn(2*k, 2*k)
        P = algebra.project(X)
        
        condition = P.T @ algebra.J + algebra.J @ P
        assert np.allclose(condition, 0, atol=1e-12)
    
    def test_dimension(self):
        #Check dimension formula: dim(sp(2k)) = k(2k+1)."
        for k in [2, 3, 4]:
            algebra = Symplectic(k)
            assert algebra.dim == k * (2 * k + 1)


class TestGetAlgebra:
    #Tests for factory function
    
    def test_get_so(self):
        algebra = get_algebra('so', 5)
        assert isinstance(algebra, SpecialOrthogonal)
        assert algebra.d == 5
    
    def test_get_sl(self):
        algebra = get_algebra('sl', 8)
        assert isinstance(algebra, SpecialLinear)
        assert algebra.d == 8
    
    def test_get_sp(self):
        algebra = get_algebra('sp', 6)  # sp(6) = sp(2*3)
        assert isinstance(algebra, Symplectic)
        assert algebra.d == 6
        assert algebra.k == 3
    
    def test_invalid_algebra(self):
        with pytest.raises(ValueError):
            get_algebra('invalid', 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
