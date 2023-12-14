# pylspack

A lightweight, multithreaded **Py**thon **pack**age for **L**everage **S**cores computations of tall-and-thin, dense or sparse matrices of arbitrary rank. Includes algorithms for:
- statistical leverage scores
- column subset selection
- sketching (CountSketch and Gaussian embeddings)
- squared row-norms computation
- basic linear algebra tools like rank-k update, diagonal scaling and more. 

This package aims to provide quick access to multithreaded algorithms with minimal memory overhead in Python, which are compatible with standard `numpy.ndarray` and `scipy.sparse.csr_matrix` data structures. 
The individual kernels can be used as a building block for various 
problems like ***low rank approximations***, ***least squares regression***, ***preconditioning***, ***feature selection***, ***clustering***, to name a few.
It is left to the user to unlock the package's full potential.

The basic matrix algorithms of this package are developed in C++, using OpenMP for multithreading and SIMD instructions.
As already noted, the implementation is designed for existing data structures of SciPy and Numpy and therefore the entire codebase is **only tested** and **should only be used** via the python wrappers.
The C++ API **can** be used as a standalone package, but it has not been tested.

### Citation
If you use this software in academic work, please consider citing the corresponding publications:
- https://doi.org/10.1137/20m1314471
```
@article{sobczyk2021estimating,
  title={Estimating leverage scores via rank revealing methods and randomization},
  author={Sobczyk, Aleksandros and Gallopoulos, Efstratios},
  journal={SIAM Journal on Matrix Analysis and Applications},
  volume={42},
  number={3},
  pages={1199--1228},
  year={2021},
  doi={10.1137/20m1314471},
  url={https://doi.org/10.1137/20m1314471},
  publisher={SIAM}
}
```
- https://doi.org/10.1145/3555370
```
@article{sobczyk2022pylspack,
  title={pylspack: Parallel Algorithms and Data Structures for Sketching, Column Subset Selection, Regression, and Leverage Scores},
  author={Sobczyk, Aleksandros and Gallopoulos, Efstratios},
  journal={ACM Transactions on Mathematical Software},
  volume={48},
  number={4},
  pages={1--27},
  year={2022},
  publisher={ACM New York, NY}
}
```

## Usage

A simple usage example to compute the leverage scores of a sparse `csr_matrix` with the `ls_via_inv_gram` method.
```python
from scipy.sparse import random
from pylspack.leverage_scores import ls_via_inv_gram
A = random(10000, 20, density=0.3, format='csr')
q = ls_via_inv_gram(A)
```

### Python API - Quick reference

#### Basic linear algebra kernels: `pylspack.linalg_kernels`

Fuction | Description
---|---
`csrcgs(A, m, r)` | Given a csr_matrix A, compute the product GSA where G is a matrix with standard normal random elements and S is a CountSketch transform (G can be omitted, in which case only SA is returned).
`csrjlt(A, m)` | Given a csr_matrix A, compute the product GA where G is as above.
`csrrk(alpha, A, beta, C)` | Given a csr_matrix A and a matrix C, computes C <- alpha * A' * A + beta * C (C is updated inplace).
`csrsqn(alpha, A, B, beta, x)` | Given a csr_matrix A and a matrix B, computes the squared Euclidean norms of the rows of the matrix C = A * B, without explicitly forming C, and store the result in the vector x (x is updated inplace).
`rmcgs(A, m, r)` | Same as csrcgs, but in this case A is a dense matrix in Row-Major (C_CONTIGUOUS) format.
`rmdsc(B, D)` | Given matrix B and a diagonal matrix D stored as a vector, computes the product B <- B * D (B is updated inplace).
`rmsqn(alpha, A, B, beta, x)` | Same as csrsqn, but in this case A is a dense matrix in Row-Major (C_CONTIGUOUS) format.
`gemm(alpha, A, B, beta, C)` | Simplified version of standard BLAS dgemm. **NOTE**: this method was implemented only to be used internally by the higher level methods, to avoid external dependencies in the C++ API. It is highly recommended to use the scipy.linalg.blas.dgemm instead, which is faster and more flexible!
`scale(B, alpha)` | Compute B <- B * alpha for some matrix B and some scalar alpha (B is updated inplace).
`set_randn(B)` | Fill the elements of a matrix B with random elements from the standard normal distribution (B is updated inplace).
`set_value(B, value)` | Fill the elements of a matrix B with fixed given float value (B is updated inplace).

#### High level algorithms: `pylspack.leverage_scores`
Function | Description
---|---
`sample_columns(A, rcond, m, r)` | Select a subset of columns of A, such that the dominant-k subspace of the subset will be close to the dominant-k subspace of A. m and r define the parameters for the underlying call to `csrcgs/rmcgs`, and rcond is used as a threshold for the small singular values to determine the numerical rank k and the subspace dimension.
`ls_via_inv_gram(A, rcond)` | Compute the leverage scores of the best rank-k approximation of A, which is determined based on rcond.
`ls_via_sketched_svd(A, rcond, m, r1, r2)` | Compute the leverage scores of the best rank-k approximation of A approximately using sketching (`rmcgs/csrcgs`). m, r1, r2 define the sketching dimensions.
`ls_hrn_exact(A, rcond, m, r)` | Compute the leverage scores of A using `sample_columns` as a first step and then calling `ls_via_inv_gram` on the selected column subset.
`ls_hrn_approx(A, rcond, m, r, m_ls, r1_ls, r2_ls)` | Compute the leverage scores of A approximately using `sample_columns` as a first step and then calling `ls_via_sketched_svd` on the selected column subset.


## Install

Requirements:
- C++11 capable compiler
- CMake >= 3.11
- OpenMP >= 4.0 (simd directives)
- Numpy and SciPy (ideally with OpenMP support)

The installation will build the C/C++ code to generate the shared library that is used by the python wrappers. ***NOTE***: In order to keep the code architecture-independent the compiler optimization flags are kept as generic as possible. In order to apply additional optimization flags, simply add them in the `${PYLSPACK_ADDITIONAL_CMAKE_CXX_FLAGS}` environment variable prior to executing pip install:
```bash
python3 -m venv venv
source venv/bin/activate
# Optional step to add more optimization flags:
# export PYLSPACK_ADDITIONAL_CMAKE_CXX_FLAGS="-march=native "
pip install git+https://github.com/IBM/pylspack
```

## Testing

To run the tests:
```bash
python3 -m pip install -r test_requirements.txt
cd test
python3 -m pytest -svvv .
# If you get an error about liblinalg_kernels.so, do the following:
# PYLSPACK_LOCATION="$(pip show pylspack | grep Location: | awk '{print $2}')/pylspack/"
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYLSPACK_LOCATION}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Tips

It is recommended to always set `OMP_NUM_THREADS`, `OMP_PLACES` and `OMP_PROC_BIND` appropriately for better performance.
For more details see the official documentation of [OpenMP](https://www.openmp.org/spec-html/5.0/openmpch6.html).
