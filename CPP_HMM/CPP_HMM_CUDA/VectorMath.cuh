#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utilities.h"

#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>

//#define CUDA_35

//---------------------------------------------------------------------------------------------------------
// CPU - serial implementation
//---------------------------------------------------------------------------------------------------------

/** Sum of elementwise multiplication of row and column vector of two two dimensional matrices.
  * Matrix layout is row mayor, matrix indexing is row first.
  * @param host_U first input matrix that provided the row
  * @param host_V second input matrix that provides the column
  * @param index_row_i row index in the first matrix U
  * @param index_column_j column index in the second matrix V
  * @param dim1_U first dimension of matrix layout in U
  * @param dim1_V first dimension of matrix layout in V
  * @return result of cross product
  */
__host__ double sumElementMulMatrixHost(const double *host_U, const double *host_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V);

/** Elementwise muliplication of row in the first and column in the second two dimensional matrix.
  * Matrix layout is row mayor, matrix indexing is row first.
  * @param host_w vector that holds the result of the elementwise multiplication, needs to be allocated first
  * @param host_U first input matrix that provided the row
  * @param host_V second input matrix that provides the column
  * @param index_row_i row index in the first matrix U
  * @param index_column_j column index in the second matrix V
  * @param dim1_U first dimension of matrix layout in U
  * @param dim1_V first dimension of matrix layout in V
  * @return result of cross product
  */
__host__ void elementMulMatrixHost(double *host_w, const double *host_U, const double *host_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V);

//---------------------------------------------------------------------------------------------------------
// GPU - parallel implementation
// e.g. with reduction
//---------------------------------------------------------------------------------------------------------

/** Sum of elementwise multiplication of row and column vector of two two dimensional matrices.
  * Matrix layout is row mayor, matrix indexing is row first.
  * @param dev_U first input matrix that provided the row
  * @param dev_V second input matrix that provides the column
  * @param index_row_i row index in the first matrix U
  * @param index_column_j column index in the second matrix V
  * @param dim1_U first dimension of matrix layout in U
  * @param dim1_V first dimension of matrix layout in V
  * @return result of cross product
  */
#ifdef CUDA_35
__device__ double sumElementMulMatrixDevice(const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V);
__device__ double sumVectorDevice(double *dev_w, unsigned int dim_w, bool destructiveSummation = false);
#else
__host__ double sumElementMulMatrixDevice(const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V);
__host__ double sumVectorDevice(double *dev_w, unsigned int dim_w, bool destructiveSummation = false);
#endif

__global__ void sumVectorKernel(double *dev_sum, const double *dev_w, unsigned int dim_sum, unsigned int dim_w);


//---------------------------------------------------------------------------------------------------------

/** Elementwise muliplication of row in the first and column in the second two dimensional matrix.
  * Matrix layout is row mayor, matrix indexing is row first.
  * @param dev_w vector that holds the result of the elementwise multiplication, needs to be allocated first
  * @param dev_U first input matrix that provided the row
  * @param dev_V second input matrix that provides the column
  * @param index_row_i row index in the first matrix U
  * @param index_column_j column index in the second matrix V
  * @param dim1_U first dimension of matrix layout in U
  * @param dim1_V first dimension of matrix layout in V
  * @return result of cross product
  */
#ifdef CUDA_35
__device__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V);
#else
__host__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V);
#endif

__global__ void elementMulMatrixKernel(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V);


//---------------------------------------------------------------------------------------------------------

__host__ void calculateAlphaTrellis3DTimeslice(double *dev_Alpha3D, const double *dev_B, const double *dev_A, unsigned int M_noOfObsSequences, unsigned int N_noOfStates, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);

//---------------------------------------------------------------------------------------------------------

// EXCERPT TAKEN FROM CUDA SAMPLE: matrixMULCUBLAS
// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

/** 
  * Wrapper function for cublas multiplication
  * @param row_A number of rows in matirx A
  * @param col_B number of columns in matrix B
  * @param col_A number of columns in matrix A
  * @param A matrix A stored on device memory 
  * @param B matrix B stored on device memory 
  * @param C result matrix C stored on device memory 
  */

__host__ void cublasMultiplyDouble(int row_A, int col_B, int col_A,const double* A_dev,const double* B_dev,double* C_dev);

