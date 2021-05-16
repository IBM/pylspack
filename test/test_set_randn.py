import numpy as np
import pytest
from pylspack.linalg_kernels import set_randn
from .utils import A_shapes_generic as A_shapes


def execute_and_check(A: np.ndarray):
    A_copy = A.copy()
    set_randn(A_copy)
    assert not np.isnan(A_copy).any()


@pytest.mark.parametrize('A_shape', [item for item in A_shapes])
def test_set_randn(A_shape):
    # set to NaN and check if it is still NaN after setting to randn
    A = np.zeros((A_shape[0], A_shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = np.nan
    execute_and_check(A)
