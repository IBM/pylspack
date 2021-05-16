import numpy as np
from scipy.sparse import random, csr_matrix
from scipy.linalg import svd
import pytest
from pylspack.leverage_scores import (
    sample_columns, ls_via_inv_gram, ls_via_sketched_svd, ls_hrn_exact, ls_hrn_approx,
    get_rank_from_vector
)
from .utils import eps_machine

density = [0.1, 0.3, 1]
matrices = [
    random(_m, _n, density=_density, format='csr')
    for _m, _n in zip([783, 1284, 4169, 6077, 9178, 13171, 17633], [2, 4, 17, 23, 31, 37, 49])
    for _density in density
]


def test_get_rank_from_vector():
    s = np.zeros((10, ))
    assert get_rank_from_vector(s, rcond=0.5) == 0
    s = np.arange(10, 0, -1)
    assert get_rank_from_vector(s, rcond=0.65) == 4
    assert get_rank_from_vector(s, rcond=0.05) == 10
    assert get_rank_from_vector(s, rcond=0) == 10
    assert get_rank_from_vector(s, rcond=1e-9) == 10
    assert get_rank_from_vector(s, rcond=float(s[-1] / s[0] - eps_machine)) == 10
    assert get_rank_from_vector(s, rcond=float(s[-1] / s[0] + eps_machine)) == 9
    assert get_rank_from_vector(s, rcond=float(s[-5] / s[0] - eps_machine)) == 6
    assert get_rank_from_vector(s, rcond=float(s[-5] / s[0] + eps_machine)) == 5
    assert get_rank_from_vector(s, rcond=0.99) == 1
    with pytest.raises(ValueError):  # raise for rcond >= 1
        _ = get_rank_from_vector(s, rcond=1.2)
    with pytest.raises(ValueError):  # raise for rcond < 0
        _ = get_rank_from_vector(s, rcond=-0.7)
    s = np.random.rand(100, )  # test raise for unsorted
    with pytest.raises(ValueError):
        _ = get_rank_from_vector(s, rcond=0.5)


def execute_and_check_sample_columns(A, m, r):
    A_res = sample_columns(A, m=m, r=r)
    assert A_res.shape[0] == A.shape[0]
    assert A_res.shape[1] <= A.shape[1]
    assert isinstance(A_res, csr_matrix)

    A_res_dense = sample_columns(A.toarray(), m=m, r=r)
    assert A_res_dense.shape[0] == A.shape[0]
    assert A_res_dense.shape[1] <= A.shape[1]
    assert A_res_dense.flags['C_CONTIGUOUS'] is True

    A_res = sample_columns(A, rcond=0, m=m, r=r)
    assert A_res.shape == A.shape
    assert isinstance(A_res, csr_matrix)

    A_res_dense = sample_columns(A.toarray(), rcond=0, m=m, r=r)
    assert A_res_dense.shape == A.shape
    assert A_res_dense.flags['C_CONTIGUOUS'] is True

    with pytest.raises(ValueError):
        _ = sample_columns(A, rcond=0, m=m, r=2 * A.shape[0])
    with pytest.raises(TypeError):  # check raise for wrong matrix type
        _ = sample_columns(random(100, 100, density=0.1))
    with pytest.raises(TypeError):  # check raise for wrong dense matrix order
        _ = sample_columns(np.ones((100, 100), order='F'))


@pytest.mark.parametrize('A', matrices)
def test_sample_columns(A):
    execute_and_check_sample_columns(A, m=None, r=None)
    execute_and_check_sample_columns(A, m=0, r=2 * A.shape[1])
    execute_and_check_sample_columns(A, m=2 * A.shape[1], r=4 * A.shape[1])


def assert_leverage_scores(q, q_true, sum, rtol=1e-10, atol=eps_machine):
    if rtol is not None:
        assert np.abs(np.sum(q) - sum) < q.shape[0] * eps_machine * sum
        assert np.allclose(q, q_true, rtol=rtol, atol=atol)
    assert np.max(np.abs(q)) <= 1
    assert np.min(q) >= 0


@pytest.mark.parametrize('A', matrices)
def test_ls_via_inv_gram(A):
    U, S, Vt = svd(A.toarray(), full_matrices=False)
    q_true = np.sum(U * U, axis=1)

    q = ls_via_inv_gram(A, rcond=0)
    assert_leverage_scores(q, q_true, sum=float(A.shape[1]))
    q = ls_via_inv_gram(A.toarray(), rcond=0)
    assert_leverage_scores(q, q_true, sum=float(A.shape[1]))

    rank_A = np.min([5, A.shape[1] - 1])
    rcond = (S[rank_A - 1] + S[rank_A]) / (2 * S[0])
    U = U[:, :rank_A]
    q_true = np.sum(U * U, axis=1)

    q = ls_via_inv_gram(A, rcond=rcond)
    assert_leverage_scores(q, q_true, sum=float(rank_A))
    q = ls_via_inv_gram(A.toarray(), rcond=rcond)
    assert_leverage_scores(q, q_true, sum=float(rank_A))
    with pytest.raises(TypeError):  # check raise for wrong matrix type
        _ = sample_columns(random(100, 100, density=0.1))
    with pytest.raises(TypeError):  # check raise for wrong dense matrix order
        _ = sample_columns(np.ones((100, 100), order='F'))


@pytest.mark.parametrize('A', matrices[6:])
def test_ls_via_sketched_svd(A):
    U, S, Vt = svd(A.toarray(), full_matrices=False)
    q_true = np.sum(U * U, axis=1)

    q = ls_via_sketched_svd(A, rcond=0, r2=0)
    assert_leverage_scores(q, q_true, rtol=0.9, sum=float(A.shape[1]))
    q = ls_via_sketched_svd(A.toarray(), rcond=0, r2=0)
    assert_leverage_scores(q, q_true, rtol=0.9, sum=float(A.shape[1]))

    rank_A = np.min([5, A.shape[1] - 1])
    U = U[:, :rank_A]
    S = S[:rank_A]
    Vt = Vt[:rank_A, :]
    A = csr_matrix(U.dot(np.diag(S)).dot(Vt))
    q_true = np.sum(U * U, axis=1)

    q = ls_via_sketched_svd(A, rcond=1e-12, r2=0)
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))
    q = ls_via_sketched_svd(A.toarray(), rcond=1e-12, r2=0)
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))
    q = ls_via_sketched_svd(A, rcond=0, r2=int(np.ceil(A.shape[1] / 2)))
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))
    q = ls_via_sketched_svd(A.toarray(), rcond=0, r2=int(np.ceil(A.shape[1] / 2)))
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))

    with pytest.raises(TypeError):  # check raise for wrong matrix type
        _ = sample_columns(random(100, 100, density=0.1))
    with pytest.raises(TypeError):  # check raise for wrong dense matrix order
        _ = sample_columns(np.ones((100, 100), order='F'))


@pytest.mark.parametrize('A', matrices)
def test_ls_hrn_exact(A):
    U, S, Vt = svd(A.toarray(), full_matrices=False)
    q_true = np.sum(U * U, axis=1)

    q = ls_hrn_exact(A, rcond=0)
    assert_leverage_scores(q, q_true, sum=float(A.shape[1]))
    q = ls_hrn_exact(A.toarray(), rcond=0)
    assert_leverage_scores(q, q_true, sum=float(A.shape[1]))

    rank_A = np.min([5, A.shape[1] - 1])
    U = U[:, :rank_A]
    S = S[:rank_A]
    Vt = Vt[:rank_A, :]
    A = csr_matrix(U.dot(np.diag(S)).dot(Vt))
    q_true = np.sum(U * U, axis=1)

    q = ls_hrn_exact(A, rcond=1e-12)
    assert_leverage_scores(q, q_true, sum=float(rank_A))
    q = ls_hrn_exact(A.toarray(), rcond=1e-12)
    assert_leverage_scores(q, q_true, sum=float(rank_A))

    with pytest.raises(TypeError):  # check raise for wrong matrix type
        _ = sample_columns(random(100, 100, density=0.1))
    with pytest.raises(TypeError):  # check raise for wrong dense matrix order
        _ = sample_columns(np.ones((100, 100), order='F'))


@pytest.mark.parametrize('A', matrices[6:])
def test_ls_hrn_approx(A):
    U, S, Vt = svd(A.toarray(), full_matrices=False)
    q_true = np.sum(U * U, axis=1)

    q = ls_hrn_approx(A, rcond=0, r2_ls=0)
    assert_leverage_scores(q, q_true, rtol=0.95, sum=float(A.shape[1]))
    q = ls_hrn_approx(A.toarray(), rcond=0, r2_ls=0)
    assert_leverage_scores(q, q_true, rtol=0.95, sum=float(A.shape[1]))

    rank_A = np.min([5, A.shape[1] - 1])
    U = U[:, :rank_A]
    S = S[:rank_A]
    Vt = Vt[:rank_A, :]
    A = csr_matrix(U.dot(np.diag(S)).dot(Vt))
    q_true = np.sum(U * U, axis=1)

    q = ls_hrn_approx(A, rcond=1e-12, r2_ls=0)
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))
    q = ls_hrn_approx(A.toarray(), rcond=1e-12, r2_ls=0)
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))
    q = ls_hrn_approx(A, rcond=0, r2_ls=int(np.ceil(A.shape[1] / 2)))
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))
    q = ls_hrn_approx(A.toarray(), rcond=0, r2_ls=int(np.ceil(A.shape[1] / 2)))
    assert_leverage_scores(q, q_true, rtol=None, sum=float(rank_A))

    with pytest.raises(TypeError):  # check raise for wrong matrix type
        _ = sample_columns(random(100, 100, density=0.1))
    with pytest.raises(TypeError):  # check raise for wrong dense matrix order
        _ = sample_columns(np.ones((100, 100), order='F'))
