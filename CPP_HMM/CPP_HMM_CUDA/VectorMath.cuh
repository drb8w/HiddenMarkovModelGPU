#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>

//---------------------------------------------------------------------------------------------------------
// CPU - serial implementation
//---------------------------------------------------------------------------------------------------------

/** Multiplication of row and column vector of two two dimensional matrices.
  * Matrix layout is row mayor, matrix indexing is row first.
  * @param host_U first input matrix that provided the row
  * @param host_V second input matrix that provides the column
  * @param index_row_i row index in the first matrix U
  * @param index_column_j column index in the second matrix V
  * @param dim1_U first dimension of matrix layout in U
  * @param dim1_V first dimension of matrix layout in V
  * @return result of cross product
  */
__host__ double rowColumnMulMatrixHost(const double *host_U, const double *host_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V);

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
__host__ void elementMulMatrixHost(double *host_w, const double *host_U, const double *host_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V);

//---------------------------------------------------------------------------------------------------------
// GPU - parallel implementation
// e.g. with reduction
//---------------------------------------------------------------------------------------------------------

/** Multiplication of row and column vector of two two dimensional matrices.
  * Matrix layout is row mayor, matrix indexing is row first.
  * @param dev_U first input matrix that provided the row
  * @param dev_V second input matrix that provides the column
  * @param index_row_i row index in the first matrix U
  * @param index_column_j column index in the second matrix V
  * @param dim1_U first dimension of matrix layout in U
  * @param dim1_V first dimension of matrix layout in V
  * @return result of cross product
  */
__device__ double rowColumnMulMatrixDevice(const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V);

__global__ void rowColumnMulMatrixKernel(double *dev_w, const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V);

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
__device__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V);

__global__ void elementMulMatrixKernel(double *dev_w, const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V);
