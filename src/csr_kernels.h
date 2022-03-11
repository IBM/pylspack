#ifndef CSR_KERNELS_H
#define CSR_KERNELS_H

extern "C" {
  /*!
   * CSR matrix sketching with CountSketch and Gaussian transforms. Parallelized with OpenMP.
   * A is a n*d CSR matrix. S is a r*n CountSketch. C is a row major matrix.
   * If m == 0: C has size r * d and we only apply C <- S * A (G is not applied).
   * If m > 0: C has size m * d. In this case C <- G * S * A * (1/sqrt(m))
   * G is a m*r matrix with elements from the standard normal distribution.
   *
   * @param d the number of columns of A and C.
   * @param m the number of rows of G. If m>0, it is also the number of rows of C.
   * @param n the number of rows of A and columns of S.
   * @param r the number of rows of S and columns of G. If m=0, r is also the number of rows of C.
   * @param A_indptr pointer to the indptr array of the CSR matrix A.
   * @param A_indices pointer to the indices array of the CSR matrix A.
   * @param A_data pointer to the data array of the CSR matrix A.
   * @param C pointer to the array of storing matrix C in row-major format.
   */
  void csrcgs( const int d, const int m, const int n, const int r, const int nnz, int *const A_indptr,
               int *const A_indices, double *const A_data, double *const C ) {
    CountSketch S( r, n, std::thread::hardware_concurrency() );
    S.populate();

    if ( m == 0 ) {
      S.apply_csr( d, 1, A_indptr, A_indices, A_data, 0, C, 0, r );
    } else {
      set_value( m, d, C, 0 );
      const int block_size = d;
      const int n_blocks = static_cast<int>( std::ceil( static_cast<double>( r ) / static_cast<double>( block_size ) ) );
      double *_G = new double[m * block_size];
      double *_T = new double[block_size * d];
      set_value( block_size, d, _T, 0 );

      for ( int i = 0; i < n_blocks - 1; ++i ) {
        set_randn( m, block_size, _G );
        S.apply_csr( d, 1, A_indptr, A_indices, A_data, 0, _T, i * block_size, ( i + 1 ) *block_size );
        gemm( m, d, block_size, 1, _G, _T, 1, C );
      }

      set_randn( m, block_size, _G );
      S.apply_csr( d, 1, A_indptr, A_indices, A_data, 0, _T, ( n_blocks - 1 ) * block_size, r );
      gemm( m, d, ( r - ( n_blocks - 1 ) *block_size ), 1, _G, _T, 1, C );
      double scale_factor = static_cast<double>( 1 ) / sqrt( static_cast<double>( m ) );
      scale( m, d, C, scale_factor );
      delete[] _T;
      delete[] _G;
    }
  }

  /*!
   * Computes the diagonal of the matrix A * B * A' and stores it in the vector x. A is a CSR
   * matrix and B is dense in row-major format. Parallelized with OpenMP.
   *
   * @param m the number of rows of A.
   * @param n the number of columns of A and rows of B.
   * @param nnz the number of non-zero elements of A.
   * @param alpha scalar to multiply ( A * B ).
   * @param A_indptr pointer to the indptr array of the CSR matrix A.
   * @param A_indices pointer to the indices array of the CSR matrix A.
   * @param A_data pointer to the data array of the CSR matrix A.
   * @param beta scalar to multiply the vector x.
   * @param B pointer to the array of storing matrix B in row-major format.
   * @param x pointer to the array of storing vector x.
   */
  void csrsqn( const int m, const int n, const int nnz, const double alpha, int *const A_indptr, int *const A_indices,
               double *const A_data, const double beta, double *const B, double *const x ) {
    int i = 0;

    if ( beta != 1 ) {
      scale( m, 1, x, beta );
    }

    if ( alpha != 0 ) {
      #pragma omp parallel shared(i)
      {
        int up, lo, ind1, ind2, _i;
        double A_ij, x_i;
        double *_B;

        #pragma omp atomic capture
        _i = i++;

        for (; _i < m; ) {
          lo = A_indptr[_i];
          up = A_indptr[_i + 1];
          x_i = 0;

          if ( lo < up ) {
            for ( ind1 = lo; ind1 < up; ++ind1 ) {
              A_ij = A_data[ind1];
              _B = & ( B[A_indices[ind1] * n] );

              for ( ind2 = lo; ind2 < up; ++ind2 ) {
                x_i += A_ij * A_data[ind2] * _B[A_indices[ind2]];
              }
            }
          }

          x[_i] += alpha * x_i;

          #pragma omp atomic capture
          _i = i++;
        }
      }
    }
  }

  /*!
   * Computes: C <- alpha * A' * A + beta * C. Parallelized with OpenMP. A' is the transpose of
   * the CSR matrix A and C is dense and stored in row-major format.
   *
   * @param m the number of rows of A.
   * @param n the number of columns of A, rows of C and columns of C.
   * @param nnz the number of non-zero elements of A.
   * @param alpha scalar to multiply ( A' * A ).
   * @param A_indptr pointer to the indptr array of the CSR matrix A.
   * @param A_indices pointer to the indices array of the CSR matrix A.
   * @param A_data pointer to the data array of the CSR matrix A.
   * @param beta scalar to multiply the matrix C.
   * @param C pointer to the array of storing matrix C in row-major format.
   */
  void csrrk( const int m, const int n, const int nnz, const double alpha, int *const A_indptr,  int *const A_indices,
              double *const A_data, const double beta, double *const C ) {
    int i, k;
    double *_C;

    if ( beta != 1 ) {
      scale( n, n, C, beta );
    }

    if ( alpha == 0 ) {
      return;
    }

    #pragma omp parallel private(i, k, _C)
    {
      const int thread_id = omp_get_thread_num();
      std::pair<int, int> limits;
      int up, lo, k_ind, j_ind;
      double A_ki;
      int block_size = static_cast<int>( std::ceil( static_cast<double>( n ) / static_cast<double>
                                         ( omp_get_num_threads() ) ) );
      limits.first  = block_size * thread_id;
      limits.second = block_size * ( thread_id + 1 );
      limits.second = std::min( limits.second, n );

      for ( k = 0; k < m; ++k ) {
        lo = A_indptr[k];
        up = A_indptr[k + 1];

        for ( k_ind = lo; k_ind < up; ++k_ind ) {
          i = A_indices[k_ind];

          if ( ( i >= limits.first ) && ( i <  limits.second ) ) {
            A_ki = alpha * A_data[k_ind];
            _C = & ( C[i * n] );

            for ( j_ind = lo; j_ind < up; ++j_ind ) {
              _C[A_indices[j_ind]] +=  A_ki * A_data[j_ind];
            }
          }
        }
      }
    }
  }

  /*!
   * Computes: C <- A' * G * (1/sqrt(m)). Parallelized with OpenMP.
   * - A' has size d*n, and is the transpose of the n*d CSR matrix A
   * - G has elements from the standard normal distribution.
   * - C has size d*m and is stored in row-major format
   *
   * @param d the number of columns of A
   * @param m the number of columns of C
   * @param n the number of rows of A
   * @param A_indptr pointer to the indptr array of the CSR matrix A
   * @param A_indices pointer to the indices array of the CSR matrix A
   * @param A_data pointer to the data array of the CSR matrix A
   * @param beta scalar to multiply the matrix C
   * @param C pointer to the array of storing matrix C in row-major format
   */
  void csrjlt( int d, int m, int n, int nnz, int *A_indptr, int *A_indices, double *A_data, double *C ) {
    set_value( m, d, C, 0 );
    double *_C;
    int i, j, k, up, lo;

    double *G = new double[m];
    #pragma omp parallel private(_C, i, j, k, up, lo)
    {
      double A_ki;
      const int thread_id = omp_get_thread_num();
      const int block_size = static_cast<int>( std::ceil( static_cast<double>( m ) / static_cast<double>
                             ( omp_get_num_threads() ) ) );
      std::pair<int, int> row_limits;
      row_limits.first = block_size * thread_id;
      row_limits.second = block_size * ( thread_id + 1 );
      row_limits.second = std::min( row_limits.second, m );
      const int n_rows = row_limits.second - row_limits.first;
      std::random_device rd{};
      std::mt19937_64 gen{rd()};
      std::normal_distribution<double> dist;

      double *_G = &( G[ row_limits.first ] );
      for ( k = 0; k < n; ++k ) {
        lo = A_indptr[k];
        up = A_indptr[k + 1];

        if ( lo < up ) {
          # pragma omp simd

          for ( j = 0; j < n_rows; ++j ) {
            _G[j] = dist( gen );
          }

          for ( i = lo; i < up; ++i ) {
            A_ki = A_data[i];
            _C = & ( C[A_indices[i] * m + row_limits.first] );

            #pragma omp simd
            for ( j = 0; j < n_rows; ++j ) {
              _C[j] += A_ki * _G[j];
            }
          }
        }
      }
    }

    delete[] G;
    double scale_factor = static_cast<double>( 1 ) / sqrt( static_cast<double>( m ) );
    scale( m, d, C, scale_factor );
  }
}
#endif
