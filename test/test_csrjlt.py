import numpy as np
from scipy.sparse import random
from scipy.linalg import svd
import pytest
from pylspack.linalg_kernels import csrjlt

m_values = list(range(113, 1100, 231))
n_values = [int(np.ceil(_m / _i)) for _i, _m in enumerate(m_values, start=2)]
density = [0.05, 0.1, 0.3]
csrjlt_matrices = [
    random(10 * _m, 10 * _n, density=_density, format='csr')
    for _m, _n in zip(m_values[1:], n_values[1:]) for _density in density[1:]
]


@pytest.mark.parametrize('A', csrjlt_matrices)
def test_csrjlt(A):
    # Test metadata
    n, d = A.shape
    gamma = 2
    m = gamma * d
    GA = csrjlt(A, m=m)
    assert GA.shape == (d, gamma * d)
    assert GA.flags['C_CONTIGUOUS'] is True

    # Check smallest singular value is approximated
    ua, sa, va = svd(A.toarray(), full_matrices=False)
    ub, sb, vb = svd(GA.T, full_matrices=False)
    # Set the value of alpha to have probability of success at least 0.99
    alpha = np.sqrt(2 * (-np.log(1 - 0.99)) / (gamma * d))
    s_min_A = sa[d - 1]
    s_min_GA = sb[d - 1]
    upper_bound = 1 + alpha + 1.0 / np.sqrt(gamma)
    lower_bound = 1 - alpha - 1.0 / np.sqrt(gamma)
    assert upper_bound * s_min_A > s_min_GA
    assert lower_bound * s_min_A < s_min_GA

    # Test the JLT property (preserving dot products / Euclidean lengths up to epsilon)
    epsilon = 0.25
    r = int(np.ceil(4 * np.log(d) / (epsilon**2 / 2 - epsilon**3 / 3)))
    GA = csrjlt(A, m=r)
    assert GA.shape[0] == d
    assert GA.shape[1] == r
    assert GA.flags['C_CONTIGUOUS'] is True

    row_norms_AT = np.sum(A.T.toarray() * A.T.toarray(), axis=1)
    assert row_norms_AT.shape[0] == d

    row_norms_GA = np.sum(GA * GA, axis=1)
    assert row_norms_GA.shape[0] == d

    max_norm_diff = np.max(np.abs(row_norms_AT - row_norms_GA) / np.abs(row_norms_AT))
    assert max_norm_diff < epsilon
