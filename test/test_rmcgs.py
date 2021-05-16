import numpy as np
from scipy.linalg import svd
import pytest
from pylspack.linalg_kernels import rmcgs

rmcgs_matrices = [
    np.random.rand(_m, _n) for _m, _n in zip([783, 4169, 9178, 17633], [2, 17, 31, 49])
]


@pytest.mark.parametrize('A', rmcgs_matrices)
def test_rmcgs(A):
    n, d = A.shape
    for gamma in [0, 2]:
        m = gamma * d
        r = 5 * (d**2 + d)
        n_rows_GSA = m or r
        GSA = rmcgs(A, m=m, r=r)
        assert GSA.shape[0] == n_rows_GSA
        assert GSA.shape[1] == d
        assert GSA.flags['F_CONTIGUOUS'] is False
        assert GSA.flags['C_CONTIGUOUS'] is True

        # Check smallest extreme singular values are approximated
        _, singvals_A, _ = svd(A, full_matrices=False)
        _, singvals_GSA, _ = svd(GSA, full_matrices=False)
        epsilon = 0.5
        upper_bound = (1 + epsilon)
        lower_bound = (1 - epsilon)
        if m > 0:
            # Set the value of alpha to have probability of success at least 0.99
            alpha = np.sqrt(2 * (-np.log(1 - 0.99)) / (m))
            upper_bound = upper_bound * (1 + alpha + 1.0 / np.sqrt(gamma))
            lower_bound = lower_bound * (1 - alpha - 1.0 / np.sqrt(gamma))
        assert upper_bound * singvals_A[0] > singvals_GSA[0]
        assert lower_bound * singvals_A[0] < singvals_GSA[0]
        assert upper_bound * singvals_A[d - 1] > singvals_GSA[d - 1]
        assert lower_bound * singvals_A[d - 1] < singvals_GSA[d - 1]
