from typing import Optional, Tuple
from scipy.sparse import random
import numpy as np

eps_machine = 5 * np.finfo(float).eps
min_size = 1
max_size = 2000
A_shapes_generic = [(1, 1), (1, 3), (3, 1), (3, 3), (17, 5), (5, 17), (237, 174), (237, 237)]
B_shapes_generic = [(1, 1), (3, 1), (1, 3), (3, 3), (5, 5), (17, 17), (174, 631), (237, 237)]
A_shapes_generic.extend([(8, 8), (64, 64), (128, 128)])
B_shapes_generic.extend([(8, 8), (64, 64), (128, 128)])
A_shapes_generic.extend(
    [
        tuple(np.random.randint(low=min_size, high=max_size, size=2))  # type: ignore
        for k in range(10)
    ]
)
B_shapes_generic.extend(
    [
        (A_shape[1], np.random.randint(low=min_size, high=max_size))
        for A_shape in A_shapes_generic[-10:]
    ]
)
alpha_beta_pairs_generic = [
    (1.0, 0.0), (0.0, 1.0), (2.7, 0.0), (0.0, 2.3), (27.6, -1.3), (0.006, 12.3)
]


def get_random_matrices(
    A_shape: Tuple, B_shape: Tuple, C_shape: Tuple, density_A: Optional[float] = None
):
    B = np.ascontiguousarray(np.random.rand(B_shape[0], B_shape[1]))

    if density_A is not None:
        A = random(int(A_shape[0]), int(A_shape[1]), density=density_A, format='csr')
    else:
        A = np.ascontiguousarray(np.random.rand(A_shape[0], A_shape[1]))

    if len(C_shape) == 2:
        C = np.ascontiguousarray(np.random.rand(C_shape[0], C_shape[1]))
    else:
        C = np.ascontiguousarray(np.random.rand(C_shape[0], ))

    return A, B, C


def set_arrays_elements_to_value(A: np.ndarray, B: np.ndarray, C: np.ndarray, value: float):
    A.fill(value)
    B.fill(value)
    C.fill(value)
