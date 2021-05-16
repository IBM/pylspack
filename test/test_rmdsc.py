import numpy as np
import pytest
from pylspack.linalg_kernels import rmdsc
from .utils import eps_machine, min_size, max_size, set_arrays_elements_to_value

B_shapes = [(1, 1), (3, 1), (1, 3), (3, 3), (17, 5), (17, 17), (237, 631), (631, 237), (237, 237)]
D_shapes = [(1, 1), (1, 1), (3, 3), (3, 3), (5, 5), (17, 17), (631, 631), (237, 237), (237, 237)]

for k in range(10):
    shape_0 = np.random.randint(low=min_size, high=max_size)
    D_shapes.append((shape_0, shape_0))
B_shapes.extend(
    [(np.random.randint(low=min_size, high=max_size), D_shape[0]) for D_shape in D_shapes[-10:]]
)


def execute_and_check(D: np.ndarray, B: np.ndarray):
    atol = eps_machine
    B_copy = B.copy()
    B_true = B.dot(np.diag(D))
    rmdsc(B_copy, D)
    assert np.max(np.abs(B_true - B_copy)) < atol


@pytest.mark.parametrize('D_shape,B_shape', [item for item in zip(D_shapes, B_shapes)])
def test_rmdsc(D_shape, B_shape):
    D = np.ascontiguousarray(np.random.rand(D_shape[0], ))
    B = np.random.rand(B_shape[0], B_shape[1])
    execute_and_check(D, B)
    set_arrays_elements_to_value(D, B, np.zeros((1, 1)), value=0)
    execute_and_check(D, B)
    set_arrays_elements_to_value(D, B, np.zeros((1, 1)), value=1)
    execute_and_check(D, B)
    set_arrays_elements_to_value(D, B, np.zeros((1, 1)), value=-12.6)
    execute_and_check(D, B)
