from typing import Optional, Union
from scipy.linalg import qr, blas, svd
import numpy as np
from scipy.sparse import csr_matrix
from .linalg_kernels import csrrk, csrsqn, csrcgs, rmcgs, rmsqn, set_randn, scale, rmdsc


def assert_matrix_type(A: Union[csr_matrix, np.ndarray]) -> None:
    if not (isinstance(A, csr_matrix) or isinstance(A, np.ndarray)):
        raise TypeError(
            'Only csr_matrix and ndarray matrix types '
            f'are supported ({type(A)} given'
        )
    if isinstance(A, np.ndarray):
        if not A.flags['C_CONTIGUOUS']:
            raise TypeError('Matrix must be a "C_CONTIGUOUS" ndarray.')


def get_rank_from_vector(S: np.ndarray, rcond: float) -> int:
    """
    Given a sorted vector S in descending order, determine the rank as minimum i such that
    S[i] <= S[0] * rcond, for a given float 0 <= rcond < 1.

    Args:
        S (np.ndarray): Vector S in descending order, e.g. singvals from scipy.linalg.svd().
        rcond (float): cutoff value for the small elements of S.
    Returns:
        int: the minimum value i such that S[i] < S[0] * rcond.
    Raises:
        ValueError: If S is not sorted in desc. order or if rcond is in [0,1).
    """
    if not np.any(S):
        return 0  # if all elements are zero, the rank is zero
    if not np.all(S[:-1] >= S[1:]):
        raise ValueError('Given vector to determine the rank must be sorted.')
    if rcond < 0 or rcond >= 1:
        raise ValueError(f'Given value for rcond={rcond} is invalid. Must be 0 <= rcond < 1.')

    rank_A = 0
    if rcond > 0:
        rank_A = int(np.argmin(S > S[0] * rcond))
    if rank_A == 0:  # zero is returned from argmin if the vector is exhausted.
        rank_A = S.shape[0]

    return rank_A


def sample_columns(
    A: Union[csr_matrix, np.ndarray],
    rcond: float = 1e-10,
    m: Optional[int] = None,
    r: Optional[int] = None
) -> Union[csr_matrix, np.ndarray]:
    """
    Reduce the matrix size with linalg_kernels.csrcgs and then use scipy.linalg.qr() to reveal its
    most important columns, based on rcond. Return the subset of selected columns.

    Arguments:
        A (Union[csr_matrix, np.ndarray]): csr_matrix or a np.ndarray (C_CONTIGUOUS).
        rcond (float): singular value tolerance (see get_rank_from_vector()). Defaults to 1e-10.
        m (int): number of rows (samples) for the Gaussian embedding. Defaults to None.
            If zero, the Gaussian embedding will not be used.
        r (int): number of rows (samples) for the CountSketch. Defaults to None.
    Returns:
        Union[csr_matrix, np.ndarray]: A matrix with the selected subset of columns of A.
    Raises:
        TypeError: if the matrix is neither a csr_matrix or a 'C_CONTIGUOUS' ndarray.
    """
    assert_matrix_type(A)
    n, d = A.shape
    m = m if m is not None else 2 * d
    r = r if r is not None else 5 * (d**2 + d)

    if isinstance(A, csr_matrix):
        GSA = csrcgs(A, m=m, r=r)
    else:
        GSA = rmcgs(A, m=m, r=r)

    R, P = qr(GSA, mode='r', pivoting=True)
    diag_R = np.abs(np.diag(R))
    rank_A = get_rank_from_vector(diag_R, rcond=rcond)
    _A = A

    if rank_A < d:
        _A = A[:, P[:rank_A]]  # Inefficient column slicing.
        if isinstance(A, np.ndarray):
            _A = np.ascontiguousarray(_A)

    return _A


def ls_via_inv_gram(A: Union[csr_matrix, np.ndarray], rcond: float = 1e-10) -> np.ndarray:
    """
    Compute the leverage scores of the best rank-k approximation of A, by explicitly forming the
    Gram matrix B = A' * A, then computing its (rank-k) SVD B = V * S^2 * V', ultimately returning
    the squared row norms of A * V * S. k is computed based on S and rcond.

    Arguments:
        A (Union[csr_matrix, np.ndarray]): csr_matrix or a np.ndarray (C_CONTIGUOUS).
        rcond (float): singular value tolerance (see get_rank_from_vector()). Defaults to 1e-10.
    Returns:
        np.ndarray: vector containing the row leverage scores of A.
    Raises:
        TypeError: if the matrix is neither a csr_matrix or a 'C_CONTIGUOUS' ndarray.
    """
    assert_matrix_type(A)
    n, d = A.shape
    B = np.zeros((A.shape[1], A.shape[1]))

    if isinstance(A, csr_matrix):
        csrrk(1.0, A, 0.0, B)
    else:
        B = blas.dgemm(1.0, A.T, A.T, 0.0, B, trans_a=False, trans_b=True, overwrite_c=True)

    V, S, _ = svd(B, overwrite_a=True)
    S = np.sqrt(S)
    rank_A = get_rank_from_vector(S, rcond=rcond)

    if rank_A < d:
        V = V[:, :rank_A]

    V = np.ascontiguousarray(V)
    rmdsc(V, 1.0 / S[:rank_A])
    q = np.zeros((n, ))

    if isinstance(A, csr_matrix):
        csrsqn(1.0, A, V, 0.0, q)
    else:
        rmsqn(1.0, A, V, 0.0, q)

    return q


def ls_via_sketched_svd(
    A: Union[csr_matrix, np.ndarray],
    rcond: float = 1e-10,
    m: Optional[int] = None,
    r1: Optional[int] = None,
    r2: Optional[int] = None
) -> np.ndarray:
    """
    Sketch the matrix to a smaller size using the pylspack sketching routines, and use its SVD
    to construct an "orthogonalizer" R, such that A*R has approximately orthonormal columns.
    To compute the row norms of A*R quickly, a "right JLT" Gaussian matrix G is used such that
    A*(R*G) has few columns, while the row norms of A*R and A*(R*G) are approximately equal.

    Arguments:
        A (Union[csr_matrix, np.ndarray]): csr_matrix or a np.ndarray (C_CONTIGUOUS).
        rcond (float): singular value tolerance (see get_rank_from_vector()). Defaults to 1e-10.
        m (int): number of rows for the Gaussian embedding of the csrcgs subroutine.
        r1 (int): number of rows for the CountSketch of the csrcgs subroutine.
        r2 (int): number of columns for the right JLT.
    Returns:
        np.ndarray: vector containing the row leverage scores of A.
    Raises:
        ValueError: if the sketched matrix has more rows than A, or if the number of JLT samples
            r2 is larger than the rank of A with respect to rcond.
        TypeError: if the matrix is neither a csr_matrix or a 'C_CONTIGUOUS' ndarray.
    """
    assert_matrix_type(A)
    n, d = A.shape
    m = m if m is not None else 10 * d
    r1 = r1 if r1 is not None else 5 * (d**2 + d)
    r2 = r2 if r2 is not None else 32 * int(np.ceil(np.log(n)))

    if isinstance(A, csr_matrix):
        GSA = csrcgs(A, m=m, r=r1)
    else:
        GSA = rmcgs(A, m=m, r=r1)

    if GSA.shape[0] <= 2 * d:
        _, S, Vt = svd(GSA, full_matrices=False)
    else:
        T = np.zeros((d, d), order='F')
        blas.dgemm(1.0, GSA.T, GSA.T, 0.0, T, trans_a=False, trans_b=True, overwrite_c=True)
        _, S, Vt = svd(T)
        S = np.sqrt(S)

    R = Vt
    rank_A = get_rank_from_vector(S, rcond=rcond)

    if rank_A < d:
        S = S[:rank_A]
        R = np.asfortranarray(R[:rank_A, :])
    R = R.T
    rmdsc(R, 1 / S)

    if r2 >= rank_A:
        raise ValueError(f'Number of JLT samples r2={r2} is larger than rank_A={rank_A}.')
    if r2 > 0:
        G = np.zeros((r2, d), order='F')
        P = np.zeros((r2, d), order='F')
        set_randn(G)
        scale(G, 1 / np.sqrt(r2))
        blas.dgemm(1.0, G, R, 0.0, P, overwrite_c=True)
        R = P.T

    q = np.zeros((n, ))

    if isinstance(A, csr_matrix):
        csrsqn(1.0, A, R, 0.0, q)
    else:
        rmsqn(1.0, A, R, 0.0, q)

    q = q * (float(rank_A) / np.sum(q))
    q = np.minimum(q, 1)
    q = np.maximum(q, 0)

    return q


def ls_hrn_exact(
    A: Union[csr_matrix, np.ndarray],
    rcond: float = 1e-10,
    m: Optional[int] = None,
    r: Optional[int] = None
) -> np.ndarray:
    """
    Computes the leverage scores of a matrix by first selecting a good subset of columns, based
    on rcond, and then calls the ls_via_inv_gram method on the selected column subset.

    Arguments:
        A (Union[csr_matrix, np.ndarray]): csr_matrix or a np.ndarray (C_CONTIGUOUS).
        rcond (float): singular value tolerance (see get_rank_from_vector()). Defaults to 1e-10.
        m (int): value of m for the sample_columns subroutine.
        r (int): value of r for the sample_columns subroutine.
    Returns:
        np.ndarray: vector containing the row leverage scores of A.
    Raises:
        TypeError: if the matrix is neither a csr_matrix or a 'C_CONTIGUOUS' ndarray.
    """
    assert_matrix_type(A)
    A_k = sample_columns(A, rcond=rcond, m=m, r=r)
    return ls_via_inv_gram(A_k, rcond=0)


def ls_hrn_approx(
    A: Union[csr_matrix, np.ndarray],
    rcond: float = 1e-10,
    m: Optional[int] = None,
    r: Optional[int] = None,
    m_ls: Optional[int] = None,
    r1_ls: Optional[int] = None,
    r2_ls: Optional[int] = None
) -> np.ndarray:
    """
    Computes the leverage scores of a matrix by first selecting a good subset of columns, based
    on rcond, and then runs the ls_via_sketched_svd method on the selected column subset.

    Arguments:
        A (Union[csr_matrix, np.ndarray]): csr_matrix or a np.ndarray (C_CONTIGUOUS).
        rcond (float): singular value tolerance (see get_rank_from_vector()). Defaults to 1e-10.
        m (int): value of m for the sample_columns subroutine.
        r (int): value of r for the sample_columns subroutine.
        m_ls (int): value of m for the ls_via_sketched_svd subroutine.
        r1_ls (int): value of r1 for the ls_via_sketched_svd subroutine.
        r2_ls (int): value of r2 for the ls_via_sketched_svd subroutine.
    Returns:
        np.ndarray: vector containing the row leverage scores of A.
    Raises:
        TypeError: if the matrix is neither a csr_matrix or a 'C_CONTIGUOUS' ndarray.
    """
    assert_matrix_type(A)
    A_k = sample_columns(A, rcond=rcond, m=m, r=r)
    return ls_via_sketched_svd(A_k, rcond=0, m=m_ls, r1=r1_ls, r2=r2_ls)
