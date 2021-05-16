#ifndef ROW_MAJOR_KERNELS_H
#define ROW_MAJOR_KERNELS_H

extern "C" {
  /*!
   * Row-Major matrix sketching with CountSketch and Gaussian transforms. Parallelized with OpenMP.
   * A is a n*d row-major matrix. S is a r*n CountSketch. C is a row major matrix.
   * If m == 0: C has size r * d and we only apply C <- S * A (G is not applied).
   * If m > 0: C has size m * d. In this case C <- G * S * A * (1/sqrt(m))
   * G is a m*r matrix with elements from the standard normal distribution.
   *
   * @param d the number of columns of A and C.
   * @param m the number of rows of C and G.
   * @param n the number of rows of A and columns of S.
   * @param r the number of rows of S and columns of G.
   * @param A pointer to the array of storing matrix A in row-major format.
   * @param C pointer to the array of storing matrix C in row-major format.
   */
  void rmcgs( const int d, const int m, const int n, const int r, double *const A, double *const C ) {
    CountSketch S( r, n, std::thread::hardware_concurrency() );
    S.populate();

    if ( m == 0 ) {
      S.apply_row_major( d, 1, A, 0, C, 0, r );
    } else {
      set_value( m, d, C, 0 );
      const int block_size = d;
      const int n_blocks = static_cast<int>( std::ceil( static_cast<double>( r ) / static_cast<double>( block_size ) ) );
      double *_G = new double[m * block_size];
      double *_T = new double[block_size * d];
      set_value( block_size, d, _T, 0 );

      for ( int i = 0; i < n_blocks - 1; ++i ) {
        set_randn( m, block_size, _G );
        S.apply_row_major( d, 1, A, 0, _T, i * block_size, ( i + 1 ) *block_size );
        gemm( m, d, block_size, 1, _G, _T, 1, C );
      }

      set_randn( m, block_size, _G );
      S.apply_row_major( d, 1, A, 0, _T, ( n_blocks - 1 ) * block_size, r );
      gemm( m, d, ( r - ( n_blocks - 1 ) *block_size ), 1, _G, _T, 1, C );
      double scale_factor = static_cast<double>( 1 ) / sqrt( static_cast<double>( m ) );
      scale( m, d, C, scale_factor );
      delete[] _T;
      delete[] _G;
    }
  }

  /*!
   * Computes the squared row norms of the matrix ( A * B ) and stores them in the vector x.
   * Parallelized with OpenMP. A and B are dense in row-major format.
   *
   * @param m the number of rows of A.
   * @param n the number of columns of B.
   * @param k the number of columns of A and rows of B.
   * @param alpha scalar to multiply ( A * B ).
   * @param A pointer to the array of storing matrix A in row-major format.
   * @param beta scalar to multiply the vector x.
   * @param B pointer to the array of storing matrix B in row-major format.
   * @param x pointer to the array of storing vector x.
   */
  void rmsqn( const int m, const int n, int k, const double alpha, double *const A, const double beta,
              double *const B, double *const x ) {
    if ( beta != 1 ) {
      scale( m, 1, x, beta );
    }

    if ( alpha == static_cast<double>( 0 ) ) {
      return;
    }

    #pragma omp parallel
    {
      int i, j, ind, start_row, end_row;
      double A_ij, x_i;
      double *_C = new double[n];
      double *_B, *_A;
      int thread_id = omp_get_thread_num();
      int n_threads = omp_get_num_threads();
      int block_size = static_cast<int>( std::ceil( static_cast<double>( m ) / static_cast<double>( n_threads ) ) );
      start_row  = block_size * thread_id;
      end_row = block_size * ( thread_id + 1 );
      end_row = std::min( end_row, m );

      for ( i = start_row; i < end_row; ++i ) {
        x_i = 0;
        _B = & ( B[0] );
        _A = &( A[i * k] );
        A_ij = _A[0];

        #pragma omp simd
        for ( j = 0; j < n; ++j ) {
          _C[j] = ( A_ij * _B[j] );
        }

        for ( ind = 1; ind < k; ++ind ) {
          A_ij = _A[ind];
          _B = & ( B[ind * n] );

          #pragma omp simd
          for ( j = 0; j < n; ++j ) {
            _C[j] += ( A_ij * _B[j] );
          }
        }

        for ( j = 0; j < n; ++j ) {
          x_i += _C[j] * _C[j];
        }

        x[i] += alpha * x_i;
      }

      delete[] _C;
    }
  }

  /*!
   * Computes: B <- B * D in parallel using OpenMP. D is a diagonal matrix
   * stored as a vector (DIA format) and B is dense in row-major format.
   *
   * @param m the number of rows of B.
   * @param n the number of columns of B and D.
   * @param B pointer to the array of storing matrix B in row-major format.
   * @param D pointer to the array of storing the diagonal matrix D.
   */
  void rmdsc( const int m, const int n, double *const D, double *const B ) {
    double *_B, * _D;
    int steps;

    #pragma omp parallel for private(_B, steps)
    for ( int i = 0; i < n; i += 8 ) {
      _D = & ( D[i] );
      steps = std::min( 8, n - i );

      for ( int j = 0; j < m; ++j ) {
        _B = & ( B[i + j * n] );

        #pragma omp simd
        for ( int k = 0; k < steps; ++k ) {
          _B[k] *= _D[k];
        }
      }
    }
  }
}
#endif
