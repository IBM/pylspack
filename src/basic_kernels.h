#ifndef BASIC_KERNELS_H
#define BASIC_KERNELS_H

#define GEMM_ROW_BLOCK_SIZE_A 16
#define GEMM_COLUMN_BLOCK_SIZE_B 128
#define GENERIC_VECTOR_SIZE 16

extern "C" {
  /*!
   * Set the entire matrix to the given value in parallel using OpenMP.
   *
   * @param m the number of rows of matrix B.
   * @param n the number of columns of matrix B.
   * @param B pointer to the matrix array.
   * @param v the value to set.
   */
  void set_value( const int m, const int n, double *const B, const double v ) {
    double *_B;
    int elements_to_compute;
    const unsigned long total_elements = static_cast<unsigned long>( m ) * static_cast<unsigned long>( n );
    const unsigned long vector_size = static_cast<unsigned long>( GENERIC_VECTOR_SIZE );

    #pragma omp parallel for private(_B, elements_to_compute)
    for ( unsigned long i = 0; i < total_elements; i += vector_size ) {
      _B = &( B[i] );
      elements_to_compute = std::min( vector_size, total_elements - i );

      #pragma omp simd
      for ( int j = 0; j < elements_to_compute; ++j ) {
        _B[j] = v;
      }
    }
  }

  /*!
   * Set the elements of the matrix to be standard normal random variables based on the mt19937_64 generator.
   *
   * @param m the number of rows of the matrix.
   * @param n the number of columns of the matrix.
   * @param G pointer to the matrix array.
   */
  void set_randn( const int m, const int n, double *const G ) {
    #pragma omp parallel
    {
      const int n_threads = omp_get_num_threads();
      const int thread_id = omp_get_thread_num();
      const int block_size = static_cast<int>( std::ceil( static_cast<double>( m ) / static_cast<double>( n_threads ) ) );
      std::pair<int, int> limits;
      limits.first  = block_size * thread_id;
      limits.second = block_size * ( thread_id + 1 );
      limits.second = std::min( limits.second, m );
      double *_G;
      std::random_device rd{};
      std::mt19937_64 gen{rd() };
      std::normal_distribution<double> dist;

      for ( int i = limits.first; i < limits.second; ++i ) {
        _G = & ( G[i * n] );

        # pragma omp simd
        for ( int j = 0; j < n; ++j ) {
          _G[j] = dist( gen );
        }
      }
    }
  }

  /*!
   * Scale the given m*n size matrix with a scalar in parallel using OpenMP.
   *
   * @param m the number of rows of the matrix.
   * @param n the number of columns of the matrix.
   * @param B pointer to the matrix array.
   * @param alpha scalar to multiply the matrix.
   */
  void scale( const int m, const int n, double *const B, const double alpha ) {
    double *_B;
    int elements_to_compute;
    const unsigned long total_elements = static_cast<unsigned long>( m ) * static_cast<unsigned long>( n );
    const unsigned long vector_size = static_cast<unsigned long>( GENERIC_VECTOR_SIZE );

    #pragma omp parallel for private(_B, elements_to_compute)
    for ( unsigned long i = 0; i < total_elements; i += vector_size ) {
      _B = &( B[i] );
      elements_to_compute = std::min( vector_size, total_elements - i );

      #pragma omp simd
      for ( int j = 0; j < elements_to_compute; ++j ) {
        _B[j] *= alpha;
      }
    }
  }

  /*!
   * Perform a matrix multiplication between a 2-by-4 block of A and a 4-by-n_rows_B block of B.
   * Computes: C <- alpha * A * B + C. Used as a subroutine of gemm.
   *
   * @param n_rows_A the number of rows of the matrix A.
   * @param n_cols_C the number of columns of the matrix C.
   * @param n_rows_B the number of columns of A and the number of rows of B.
   * @param alpha scalar to multiply (A*B).
   * @param A pointer to the array of storing matrix A.
   * @param LDA leading dimension of array A.
   * @param B pointer to the array of storing matrix B.
   * @param LDB leading dimension of array B.
   * @param C pointer to the array of storing matrix C.
   * @param LDC leading dimension of array C.
   */
  inline void small_gemm_2_by_4( const int n_rows_A, const int n_cols_C, const int n_rows_B, const double alpha,
                                 double *const A, const int LDA, double *const B, const int LDB, double *const C, const int LDC ) {
    double *_A1 = &( A[0] ), *_A2 = &( A[0] );
    double *_B1 = &( B[0] ), *_B2 = &( B[0] ), *_B3 = &( B[0] ), *_B4 = &( B[0] );
    double *_C1 = &( C[0] ), *_C2 = &( C[0] );
    double v11 = 0, v12 = 0, v13 = 0, v14 = 0, v21 = 0, v22 = 0, v23 = 0, v24 = 0;
    double b1, b2, b3, b4;

    if ( alpha == 0 ) {
      return;
    }

    v11 = _A1[0] * alpha;

    if ( n_rows_A == 2 ) {
      _A2 = & ( A[LDA] );
      _C2 = & ( C[LDC] );
      v21 = _A2[0] * alpha;
    }

    if ( n_rows_B > 1 ) {
      _B2 = & ( B[LDB] );
      v12 = _A1[1] * alpha;

      if ( n_rows_A == 2 ) {
        v22 = _A2[1] * alpha;
      }
    }

    if ( n_rows_B > 2 ) {
      _B3 = & ( B[2 * LDB] );
      v13 = _A1[2] * alpha;

      if ( n_rows_A == 2 ) {
        v23 = _A2[2] * alpha;
      }
    }

    if ( n_rows_B > 3 ) {
      _B4 = & ( B[3 * LDB] );
      v14 = _A1[3] * alpha;

      if ( n_rows_A == 2 ) {
        v24 = _A2[3] * alpha;
      }
    }

    if ( n_rows_A == 1 ) {

      #pragma omp simd
      for ( int h = 0; h < n_cols_C; ++h ) {
        b1 = _B1[h];
        b2 = _B2[h];
        b3 = _B3[h];
        b4 = _B4[h];
        _C1[h] +=  b1 * v11 + b2 * v12 + b3 * v13 + b4 * v14;
      }
    } else if ( n_rows_A == 2 ) {

      #pragma omp simd
      for ( int h = 0; h < n_cols_C; ++h ) {
        b1 = _B1[h];
        b2 = _B2[h];
        b3 = _B3[h];
        b4 = _B4[h];
        _C1[h] +=  b1 * v11 + b2 * v12 + b3 * v13 + b4 * v14;
        _C2[h] +=  b1 * v21 + b2 * v22 + b3 * v23 + b4 * v24;
      }
    }
  }

  /*!
   * Matrix multiplication in row-major format similar to BLAS DGEMM routine. Parallelized with OpenMP.
   * Computes: C <- alpha * A * B + beta * C
   *
   * @param m the number of rows of the matrix A.
   * @param n the number of columns of the matrix C.
   * @param k the number of columns of A and the number of rows of B.
   * @param alpha scalar to multiply (A*B).
   * @param A pointer to the array of storing matrix A.
   * @param B pointer to the array of storing matrix B.
   * @param beta scalar to multiply the matrix C.
   * @param C pointer to the array of storing matrix C.
   */
  void gemm( const int m, const int n, const int k, const double alpha, double *const A, double *const B,
             const double beta, double *const C ) {
    if ( beta != 1 ) {
      scale( m, n, C, beta );
    }

    if ( alpha == 0 ) {
      return;
    }

    double *_A, *_B, *_C;
    int n_rows_to_compute_A, n_rows_to_compute_B, n_cols_to_compute_C;
    int g, h, i, j;

    #pragma omp parallel for private(_A, _B, _C, n_rows_to_compute_A, n_rows_to_compute_B, n_cols_to_compute_C, g, h, i, j)
    for ( i = 0; i < m; i += GEMM_ROW_BLOCK_SIZE_A ) {
      for ( j = 0; j < k; j += 4 ) {
        n_rows_to_compute_B = std::min( 4, k - j );
        _B = & ( B[j * n] );

        for ( g = 0; g < n; g += GEMM_COLUMN_BLOCK_SIZE_B, _B += GEMM_COLUMN_BLOCK_SIZE_B ) {
          n_cols_to_compute_C = std::min( GEMM_COLUMN_BLOCK_SIZE_B, n - g );

          for ( h = 0; ( h < GEMM_ROW_BLOCK_SIZE_A ) && ( m - i - h ) > 0; h += 2 ) {
            _A = & ( A[( i + h ) * k + j] );
            _C = & ( C[( i + h ) * n + g] );
            n_rows_to_compute_A = std::min( 2, m - i - h );
            small_gemm_2_by_4( n_rows_to_compute_A, n_cols_to_compute_C, n_rows_to_compute_B, alpha, _A, k, _B, n, _C, n );
          }
        }
      }
    }
  }

}
#endif
