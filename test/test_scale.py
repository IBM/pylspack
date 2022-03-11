import numpy as np
import pytest
from pylspack.linalg_kernels import scale
from utils import eps_machine
from utils import A_shapes_generic as A_shapes


def execute_and_check(A: np.ndarray, alpha: float):
    atol = eps_machine
    A_copy = A.copy()
    A_true = alpha * A
    scale(A_copy, alpha)
    assert np.max(np.abs(A_true - A_copy)) < atol


def run_all(A):
    for alpha in [0.0, 1.0, 47.327, -394.6512, 0.0007635, -0.0065]:
        execute_and_check(A, alpha)


@pytest.mark.parametrize('A_shape', [item for item in A_shapes])
def test_scale(A_shape):
    A = np.ascontiguousarray(np.random.rand(A_shape[0], A_shape[1]))
    run_all(A)
    A.fill(0)
    run_all(A)
    A.fill(1)
    run_all(A)
