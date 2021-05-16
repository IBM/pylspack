#ifndef COUNTSKETCH_H
#define COUNTSKETCH_H

extern "C" {
  /*!
   * Class representing a CountSketch linear transform. The transform is defined as a sparse
   * matrix of size r*n with only one non-zero element per column. It is stored in memory as a
   * 3-level vector (vector of vectors of vectors). The top level vector, called rows_vector,
   * represents the rows of the matrix. Each element of this vector represents a single row,
   * and each row is itself a vector. To enable multithreading during the "population" of the
   * matrix, we slice the matrix horizontally in column blocks such that each thread can
   * populate a block of columns avoiding collisions and synchronization. Therefore, each row
   * vector consists of n_col_blocks vectors, each one representing a column block. Finally,
   * each column block is itself a vector of integers. Each integer is a column index. If the
   * sign of the column index is positive, then it means that at this column the value of the
   * matrix is +1, otherwise if its negative it means the value is -1.
   */
  class CountSketch {

   public:
    void print();
    void populate();
    void apply_csr( const int n_cols_output, const double alpha, const int *const A_indptr, const int *const A_indices,
                    const double *const A_data, const double beta, double *const C, const int start_row, const int end_row );
    void apply_row_major( const int n_cols_output, const double alpha, double *const A, const double beta, double *const C,
                          const int start_row, const int end_row );
    CountSketch( const int n_rows, const int n_cols, const int n_col_blocks );

   private:
    int n_rows; // number of rows of S, equals to the number of elements of rows_vector as well
    int n_cols; // number of columns of S
    int n_col_blocks; // number of column slices of S
    int col_block_size; // number of columns in each column block
    typedef std::vector<std::vector<std::vector<int>>> countsketch_rows_t;
    countsketch_rows_t rows_vector;
  };

  CountSketch::CountSketch( const int n_rows, const int n_cols, const int n_col_blocks ) : n_rows( n_rows ),
    n_cols( n_cols ),
    n_col_blocks( n_col_blocks ), rows_vector( n_rows ) {
    for ( int i = 0; i < n_rows; ++i ) {
      rows_vector[i] = std::vector<std::vector<int>> ( n_col_blocks );
    }

    col_block_size = static_cast<int>( std::ceil( static_cast<double>( n_cols ) / static_cast<double>( n_col_blocks ) ) );
  }

  /*!
   * Populate an instance of a CountSketch matrix in parallel using OpenMP. Each thread populates
   * one or a few column slices of the matrix. For each slice/block, it iterates over all column
   * indices for that block, and for each column it assigns at a random position a random sign
   * value. Since the column blocks are non overlapping, there is no need for synchronization.
   */
  void CountSketch::populate() {
    #pragma omp parallel
    {
      std::random_device rd;
      std::mt19937 int_gen( rd() );
      std::uniform_int_distribution<> int_dist( 0, n_rows - 1 );
      std::default_random_engine bool_gen;
      std::bernoulli_distribution bool_dist( 0.5 );
      const int thread_id = omp_get_thread_num();
      const int n_threads = omp_get_num_threads();
      const int blocks_per_thread = static_cast<int>( std::ceil( static_cast<double>( n_col_blocks ) / static_cast<double>( n_threads ) ) );
      const int start_block = thread_id * blocks_per_thread;
      const int end_block = std::min( ( thread_id + 1 ) * blocks_per_thread, n_col_blocks );

      for ( int block = start_block; block < end_block; ++block ) {
        int lo = block * col_block_size;
        int up = std::min( ( block + 1 ) * col_block_size, n_cols );

        for ( int k = lo; k < up; ++k ) {
          if ( bool_dist( bool_gen ) ) {
            rows_vector[int_dist( int_gen )][block].push_back( k );
          } else {
            rows_vector[int_dist( int_gen )][block].push_back( -k );
          }
        }
      }
    }
  }

  /*!
   * Computes: C <- alpha * S[start_row:end_row] * A + beta * C
   * Multiply a set of rows of S with a CSR matrix A in parallel using OpenMP. Matrix C is assumed
   * to be stored in row-major order and has size (end_row-start_row) * n_cols_output.
   *
   * @param n_cols_output the number of columns of A and C.
   * @param alpha scalar to multiply ( S * A ).
   * @param A_indptr pointer to the indptr array of the CSR matrix A.
   * @param A_indices pointer to the indices array of the CSR matrix A.
   * @param A_data pointer to the data array of the CSR matrix A.
   * @param beta scalar to multiply the matrix C.
   * @param C pointer to the array of storing matrix C in row-major format.
   * @param start_row the index of the first row of S to multiply A.
   * @param end_row the index of the last row of S to multiply A.
   */
  void CountSketch::apply_csr( const int n_cols_output, const double alpha, const int *const A_indptr,
                               const int *const A_indices,
                               const double *const A_data,
                               const double beta, double *const C, const int start_row = 0, const int end_row = 0 ) {
    int n_rows_to_apply;

    if ( ( end_row == 0 ) || ( end_row > n_rows ) ) {
      n_rows_to_apply = n_rows - start_row;
    } else {
      n_rows_to_apply = end_row - start_row;
    }

    if ( beta != 1 ) {
      scale( n_rows_to_apply, n_cols_output, C, beta );
    }

    if ( ( alpha == 0 ) || ( n_rows_to_apply <= 0 ) ) {
      return;
    }

    double v;
    int col_value, col_index;
    unsigned long C_index;
    double *_C;

    #pragma omp parallel for private(_C, col_value, col_index, v, C_index)
    for ( int k = start_row; k < end_row; ++k ) {
      C_index = static_cast<unsigned long>( k - start_row ) * static_cast<unsigned long>( n_cols_output );
      _C = & ( C[C_index] );

      for ( auto block_itr = rows_vector[k].begin(); block_itr != rows_vector[k].end(); ++block_itr ) {
        for ( auto col_itr = block_itr->begin(); col_itr != block_itr->end(); ++col_itr ) {
          col_value = *col_itr;
          col_index = abs( col_value );

          if ( col_value > 0 ) {
            v = alpha;
          } else {
            v = -alpha;
          }

          for ( int h = A_indptr[col_index]; h < A_indptr[col_index + 1]; ++h ) {
            _C[A_indices[h]] += v * A_data[h];
          }
        }
      }
    }
  }

  /*!
   * Computes: C <- alpha * S[start_row:end_row] * A + beta * C
   * Multiply a set of rows of S with a row-major dense matrix A in parallel using OpenMP.
   * Matrix C is assumed to be stored in row-major order and has size
   * (end_row-start_row) * n_cols_output.
   *
   * @param n_cols_output the number of columns of A and C.
   * @param alpha scalar to multiply ( S * A ).
   * @param A pointer to the data array of the matrix A.
   * @param beta scalar to multiply the matrix C.
   * @param C pointer to the array of storing matrix C in row-major format.
   * @param start_row the index of the first row of S to multiply A.
   * @param end_row the index of the last row of S to multiply A.
   */
  void CountSketch::apply_row_major( const int n_cols_output, const double alpha, double *const A, const double beta,
                                     double *const C,
                                     const int start_row = 0, const int end_row = 0 ) {
    int n_rows_to_apply;

    if ( ( end_row == 0 ) || ( end_row > n_rows ) ) {
      n_rows_to_apply = n_rows - start_row;
    } else {
      n_rows_to_apply = end_row - start_row;
    }

    if ( beta != 1 ) {
      scale( n_rows_to_apply, n_cols_output, C, beta );
    }

    if ( ( alpha == 0 ) || ( n_rows_to_apply <= 0 ) ) {
      return;
    }

    double v;
    int col_value, col_index;
    unsigned long A_index, C_index;
    double *_A, * _C;

    #pragma omp parallel for private(_A, _C, col_value, col_index, v, C_index)
    for ( int k = 0; k < n_rows_to_apply; ++k ) {
      C_index = static_cast<unsigned long>( k ) * static_cast<unsigned long>( n_cols_output );
      _C = & ( C[C_index] );

      for ( auto block_itr = rows_vector[k + start_row].begin(); block_itr != rows_vector[k + start_row].end();
            ++block_itr ) {
        for ( auto col_itr = block_itr->begin(); col_itr != block_itr->end(); ++col_itr ) {
          col_value = *col_itr;
          col_index = abs( col_value );

          if ( col_value > 0 ) {
            v = alpha;
          } else {
            v = -alpha;
          }

          A_index = static_cast<unsigned long>( col_index ) * static_cast<unsigned long>( n_cols_output );
          _A = & ( A[A_index] );

          for ( int h = 0; h < n_cols_output; ++h ) {
            _C[h] += v * _A[h];
          }
        }
      }
    }
  }

  /*!
   * Print S in standard output
   */
  void CountSketch::print() {
    int *arr = new int[n_rows * n_cols];
    std::fill( arr, arr + n_rows * n_cols - 1, 0 );

    for ( int i = 0; i < n_rows; ++i ) {
      for ( auto block_itr = rows_vector[i].begin(); block_itr != rows_vector[i].end(); ++block_itr ) {
        for ( auto col_itr = block_itr->begin(); col_itr != block_itr->end(); ++col_itr ) {
          int j = *col_itr;

          if ( j > 0 ) {
            arr[i * n_cols + abs( j )] = 1 ;
          } else {
            arr[i * n_cols + abs( j )] = 2 ;
          }
        }
      }
    }

    for ( int i = 0; i < n_rows; ++i ) {
      std::cout << " | ";

      for ( int j = 0; j < n_cols; ++j ) {
        char s = ' ';

        if ( arr[i * n_cols + j] == 1 ) {
          s = '+';
        } else if ( arr[i * n_cols + j] == 2 ) {
          s = '-';
        }

        std::cout << s << " | ";
      }

      std::cout << std::endl;
    }

    delete[] arr;
  }
}
#endif
