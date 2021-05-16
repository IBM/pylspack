import numpy as np
from scipy.sparse import csr_matrix
import pytest
from pylspack.linalg_kernels import csrrk
from .utils import (
    eps_machine, min_size, max_size, alpha_beta_pairs_generic, get_random_matrices,
    set_arrays_elements_to_value
)

A_shapes = [(1, 1), (3, 1), (3, 3), (17, 5), (237, 174), (237, 237)]
A_shapes.extend([np.random.randint(low=min_size, high=max_size, size=2) for k in range(10)])


def execute_and_check(alpha: float, A: csr_matrix, beta: float, C: np.ndarray):
    atol = eps_machine * A.shape[0] * A.shape[0]
    if np.max(np.abs(A)) > 0:
        atol = atol * (np.max(np.abs(A))**2)
    C_copy = C.copy()
    C_true = alpha * A.T.dot(A) + beta * C_copy
    csrrk(alpha, A, beta, C_copy)
    # matrices are scaled, we must scale also the atol
    assert np.max(np.abs(C_true - C_copy)) < atol * (np.abs(alpha) + np.abs(beta))


def run_all(A, C):
    for alpha, beta in alpha_beta_pairs_generic:
        execute_and_check(alpha, A, beta, C)


@pytest.mark.parametrize('A_shape', [item for item in A_shapes])
def test_csrrk(A_shape):
    density = np.min([5.0 / A_shape[1], 0.6])
    A, _, C = get_random_matrices(A_shape, (1, 1), (A_shape[1], A_shape[1]), density)
    run_all(A, C)
    set_arrays_elements_to_value(A.data, np.zeros((1, 1)), C, value=0)
    run_all(A, C)
    set_arrays_elements_to_value(A.data, np.zeros((1, 1)), C, value=1)
    run_all(A, C)
