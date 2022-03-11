import numpy as np
import pytest
from pylspack.linalg_kernels import gemm
from utils import (
    eps_machine, alpha_beta_pairs_generic, get_random_matrices, set_arrays_elements_to_value
)
from utils import A_shapes_generic as A_shapes
from utils import B_shapes_generic as B_shapes


def execute_and_check(alpha: float, A: np.ndarray, B: np.ndarray, beta: float, C: np.ndarray):
    atol = eps_machine * A.shape[1] * A.shape[1]
    if np.max(np.abs(A)) * np.max(np.abs(B)) > 0:
        atol = atol * np.max(np.abs(A)) * np.max(np.abs(B))
    C_copy = C.copy()
    C_true = alpha * A.dot(B) + beta * C_copy
    gemm(alpha, A, B, beta, C_copy)
    # matrices are scaled, we must scale also the bound
    assert np.max(np.abs(C_true - C_copy)) < atol * (np.abs(alpha) + np.abs(beta))


def run_all(A, B, C):
    for alpha, beta in alpha_beta_pairs_generic:
        execute_and_check(alpha, A, B, beta, C)


@pytest.mark.parametrize('A_shape,B_shape', [item for item in zip(A_shapes, B_shapes)])
def test_gemm(A_shape, B_shape):
    A, B, C = get_random_matrices(A_shape, B_shape, (A_shape[0], B_shape[1]))
    run_all(A, B, C)
    set_arrays_elements_to_value(A, B, C, value=0)
    run_all(A, B, C)
    set_arrays_elements_to_value(A, B, C, value=0)
    run_all(A, B, C)
