import os
import glob
from ctypes import CDLL, c_int, c_void_p, c_double
from typing import Optional
import numpy as np
from scipy.linalg import blas
from scipy.sparse import csr_matrix

libdir = os.path.dirname(os.path.realpath(__file__))
libfile = glob.glob(f'{libdir}/liblinalg_kernels*')
if libfile:
    ext_lib = CDLL(os.path.join(libdir, libfile[0]))
else:
    print(f'Warning: could not find {libdir}/liblinalg_kernels*')
    try:
        print('Trying to fild liblinalg_kernels.so from LD_LIBRARY_PATH...')
        ext_lib = CDLL('liblinalg_kernels.so')
    except Exception:
        print('Trying to fild liblinalg_kernels.dylib from LD_LIBRARY_PATH...')
        ext_lib = CDLL('liblinalg_kernels.dylib')

# arg types
ext_lib.csrcgs.argtypes = [
    c_int, c_int, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p
]
ext_lib.csrjlt.argtypes = [c_int, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p]
ext_lib.csrrk.argtypes = [
    c_int, c_int, c_int, c_double, c_void_p, c_void_p, c_void_p, c_double, c_void_p
]
ext_lib.csrsqn.argtypes = [
    c_int, c_int, c_int, c_double, c_void_p, c_void_p, c_void_p, c_double, c_void_p, c_void_p
]
ext_lib.gemm.argtypes = [c_int, c_int, c_int, c_double, c_void_p, c_void_p, c_double, c_void_p]
ext_lib.rmcgs.argtypes = [c_int, c_int, c_int, c_int, c_void_p, c_void_p]
ext_lib.rmdsc.argtypes = [c_int, c_int, c_void_p, c_void_p]
ext_lib.rmsqn.argtypes = [c_int, c_int, c_int, c_double, c_void_p, c_double, c_void_p, c_void_p]
ext_lib.scale.argtypes = [c_int, c_int, c_void_p, c_double]
ext_lib.set_randn.argtypes = [c_int, c_int, c_void_p]
ext_lib.set_value.argtypes = [c_int, c_int, c_void_p, c_double]

# return types
for kernel in [
    ext_lib.csrcgs, ext_lib.csrjlt, ext_lib.csrrk, ext_lib.csrsqn, ext_lib.gemm, ext_lib.rmcgs,
    ext_lib.rmdsc, ext_lib.rmsqn, ext_lib.scale, ext_lib.set_randn, ext_lib.set_value
]:
    kernel.restype = None


def assert_shape(a: int, b: int) -> None:
    if a != b:
        raise ValueError('dimension mismatch: {} is not equal to {}.'.format(a, b))


def assert_dtype(A: np.ndarray, dtype: str) -> None:
    if A.dtype != dtype:
        raise TypeError('unsupported dtype: {}.'.format(A.dtype))


def assert_contiguous_type(A: np.ndarray, contiguous_type: str) -> None:
    if A.flags[contiguous_type] is False:
        raise TypeError('array is not {} as expected.'.format(contiguous_type))


def csrrk(alpha: float, A: csr_matrix, beta: float, C: np.ndarray) -> None:
    """
    Compute the product: C <- alpha * A' * A + beta * C, where A' is the transpose of A.

    Args:
         alpha (float): scalar to multiply the product A' * A.
         A (csr_matrix): matrix A in csr format.
         beta (float): scalar to multiply C before adding to alpha * A' * A.
         C (np.ndarray): matrix C in row-major ordering (C_CONTIGUOUS).
    """
    assert_dtype(A.data, 'float64')
    assert_dtype(C, 'float64')
    assert_shape(A.shape[1], C.shape[0])
    assert_contiguous_type(C, 'C_CONTIGUOUS')
    ext_lib.csrrk(
        int(A.shape[0]), int(A.shape[1]), int(A.nnz), c_double(alpha),
        A.indptr.ctypes.data_as(c_void_p), A.indices.ctypes.data_as(c_void_p),
        A.data.ctypes.data_as(c_void_p), c_double(beta), C.ctypes.data_as(c_void_p)
    )


def rmsqn(alpha: float, A: np.ndarray, B: np.ndarray, beta: float, x: np.ndarray) -> None:
    """
    Compute the product: x <- alpha * squared_row_norms(A * B) + beta * x.

    Args:
         alpha (float): scalar to multiply the squared row norms of the product A*B.
         A (np.ndarray): matrix A in row-major ordering (C_CONTIGUOUS).
         B (np.ndarray): matrix B in row-major ordering (C_CONTIGUOUS).
         beta (float): scalar to multiply x before adding to the squared row norms of A*B.
         x (np.ndarray): vector x.
    """
    assert_dtype(A, 'float64')
    assert_dtype(B, 'float64')
    assert_shape(A.shape[1], B.shape[0])
    assert_contiguous_type(A, 'C_CONTIGUOUS')
    assert_contiguous_type(B, 'C_CONTIGUOUS')
    ext_lib.rmsqn(
        int(A.shape[0]), int(B.shape[1]), int(A.shape[1]), c_double(alpha),
        A.ctypes.data_as(c_void_p), c_double(beta), B.ctypes.data_as(c_void_p),
        x.ctypes.data_as(c_void_p)
    )


def csrsqn(alpha: float, A: csr_matrix, B: np.ndarray, beta: float, x: np.ndarray) -> None:
    """
    Compute the product: x <- alpha * squared_row_norms(A * B) + beta * x.

    Args:
         alpha (float): scalar to multiply the squared row norms of the product A*B.
         A (csr_matrix): matrix A in csr format.
         B (np.ndarray): matrix B in row-major ordering (C_CONTIGUOUS).
         beta (float): scalar to multiply x before adding to the squared row norms of A*B.
         x (np.ndarray): vector x
    """
    assert_dtype(A.data, 'float64')
    assert_dtype(B, 'float64')
    assert_shape(A.shape[1], B.shape[0])
    assert_contiguous_type(B, 'C_CONTIGUOUS')
    B = blas.dgemm(1.0, B.T, B.T, 0.0, trans_a=True, trans_b=False)
    ext_lib.csrsqn(
        int(A.shape[0]), int(B.shape[0]), int(A.nnz), c_double(alpha),
        A.indptr.ctypes.data_as(c_void_p), A.indices.ctypes.data_as(c_void_p),
        A.data.ctypes.data_as(c_void_p), c_double(beta), B.ctypes.data_as(c_void_p),
        x.ctypes.data_as(c_void_p)
    )


def rmdsc(B: np.ndarray, D: np.ndarray) -> None:
    """
    Compute B <- B * D where B is C_CONTIGUOUS and D is diagonal, updating B in-place.

    Args:
         B (np.ndarray): matrix B in row-major ordering (C_CONTIGUOUS)
         D (np.ndarray): Diagonal matrix represented as a 1-dimensional np.ndarray (a vector)
    """
    assert_dtype(D, 'float64')
    assert_dtype(B, 'float64')
    assert_shape(B.shape[1], D.shape[0])
    assert_contiguous_type(B, 'C_CONTIGUOUS')
    ext_lib.rmdsc(
        int(B.shape[0]), int(B.shape[1]), D.ctypes.data_as(c_void_p), B.ctypes.data_as(c_void_p)
    )


def csrjlt(A: csr_matrix, m: int) -> np.ndarray:
    """
    Compute the product: B <- A' * G' where G has size m * n and elements from the standard
    normal distribution, rescaled by 1/sqrt(m). The matrix G is not explicitly formed, and
    its elements are computed on-the-fly only when required.

    Args:
         A (csr_matrix): matrix A in csr format.
         m (int): number of rows for the matrix G.
    Returns:
         B (np.ndarray): matrix B in row-major ordering (C_CONTIGUOUS).
    """
    assert_dtype(A.data, 'float64')
    B = np.zeros((A.shape[1], m))
    ext_lib.csrjlt(
        int(A.shape[1]), m, int(A.shape[0]), int(A.nnz), A.indptr.ctypes.data_as(c_void_p),
        A.indices.ctypes.data_as(c_void_p), A.data.ctypes.data_as(c_void_p),
        B.ctypes.data_as(c_void_p)
    )
    return B


def csrcgs(A: csr_matrix, m: Optional[int] = None, r: Optional[int] = None) -> np.ndarray:
    """
    Compute the product: B <- G * S * A where G has size m * r and elements from the standard
    normal distribution, rescaled by 1/sqrt(m), and S is a CountSketch of size r * n. The matrix
    G is not explicitly formed, and its elements are computed on-the-fly only when required.

    Args:
         A (csr_matrix): matrix A in csr format.
         m (int): number of rows for the Gaussian sketch. If zero, the Gaussian sketch will
            not be applied. If None, it defaults to 2*d..
         r (int): number of rows to use for the CountSketch transform. If zero, this transform
            will not be applied. If None, it defaults to 5 * (d**2 + d).
    Returns:
         np.ndarray: matrix B in row-major ordering (C_CONTIGUOUS).
    """
    assert_dtype(A.data, 'float64')
    n = int(A.shape[0])
    d = int(A.shape[1])
    r = r if r is not None else 5 * (d**2 + d)
    m = m if m is not None else 2 * d
    if (m > n) or (r > n):
        raise ValueError(f'Either m={m} or r={r} is larger than n={n}, the number of rows of A.')
    n_rows_B = m or r  # if m is zero, fallback to r
    B = np.zeros((n_rows_B, d))
    ext_lib.csrcgs(
        d, m, n, r, int(A.nnz), A.indptr.ctypes.data_as(c_void_p),
        A.indices.ctypes.data_as(c_void_p), A.data.ctypes.data_as(c_void_p),
        B.ctypes.data_as(c_void_p)
    )
    return B


def rmcgs(A: np.ndarray, m: Optional[int] = None, r: Optional[int] = None) -> np.ndarray:
    """
    Compute the product: B <- G * S * A where G has size m * r and elements from the standard
    normal distribution, rescaled by 1/sqrt(m), and S is a CountSketch of size r * n. The matrix
    G is not explicitly formed, and its elements are computed on-the-fly only when required.

    Args:
         A (np.ndarray): matrix A in row-major ordering (C_CONTIGUOUS)
         m (int): number of rows for the Gaussian sketch. If zero, the Gaussian sketch will
            not be applied. If None, it defaults to 2*d.
         r (int): number of rows to use for the CountSketch transform. If zero, this transform
            will not be applied. If None, it defaults to 5 * (d**2 + d).
    Returns:
         np.ndarray: matrix B in row-major ordering (C_CONTIGUOUS)
    """
    assert_dtype(A, 'float64')
    assert_contiguous_type(A, 'C_CONTIGUOUS')
    n = int(A.shape[0])
    d = int(A.shape[1])
    r = r if r is not None else 5 * (d**2 + d)
    m = m if m is not None else 2 * d
    if (m > n) or (r > n):
        raise ValueError(f'Either m={m} or r={r} is larger than n={n}, the number of rows of A.')
    n_rows_B = m or r  # if m is zero, fallback to r
    B = np.zeros((n_rows_B, d))
    ext_lib.rmcgs(d, m, n, r, A.ctypes.data_as(c_void_p), B.ctypes.data_as(c_void_p))
    return B


def set_value(B: np.ndarray, value: float) -> None:
    """
    Set all the elements of B equal to the given value.

    Args:
         B (np.ndarray): matrix B.
         value (float): the value to set to all the elements of B.
    """
    assert_dtype(B, 'float64')
    ext_lib.set_value(int(B.shape[0]), int(B.shape[1]), B.ctypes.data_as(c_void_p), value)


def set_randn(B: np.ndarray) -> None:
    """
    Fill B with random elements from the standard normal distribution.

    Args:
         B (np.ndarray): matrix B.
    """
    assert_dtype(B, 'float64')
    ext_lib.set_randn(int(B.shape[0]), int(B.shape[1]), B.ctypes.data_as(c_void_p))


def scale(B: np.ndarray, alpha: float) -> None:
    """
    Scale the matrix B in-place by the scalar alpha: B <- alpha * B.

    Args:
         B (np.ndarray): matrix B.
         alpha (float): scalar to multiply B.
    """
    assert_dtype(B, 'float64')
    ext_lib.scale(int(B.shape[0]), int(B.shape[1]), B.ctypes.data_as(c_void_p), c_double(alpha))


def gemm(alpha: float, A: np.ndarray, B: np.ndarray, beta: float, C: np.ndarray) -> None:
    """
    Compute the product: C <- alpha * A * B + beta * C.

    Args:
         alpha (float): scalar to multiply the product A*B.
         A (np.ndarray): matrix A in row-major ordering (C_CONTIGUOUS).
         B (np.ndarray): matrix B in row-major ordering (C_CONTIGUOUS).
         beta (float): scalar to multiply C before adding to alpha * A * B.
         C (np.ndarray): matrix C in row-major ordering (C_CONTIGUOUS).
    """
    assert_shape(A.shape[0], C.shape[0])
    assert_shape(A.shape[1], B.shape[0])
    assert_shape(B.shape[1], C.shape[1])
    assert_contiguous_type(A, 'C_CONTIGUOUS')
    assert_contiguous_type(B, 'C_CONTIGUOUS')
    assert_contiguous_type(C, 'C_CONTIGUOUS')
    ext_lib.gemm(
        int(A.shape[0]), int(B.shape[1]), int(A.shape[1]), c_double(alpha),
        A.ctypes.data_as(c_void_p), B.ctypes.data_as(c_void_p), c_double(beta),
        C.ctypes.data_as(c_void_p)
    )
