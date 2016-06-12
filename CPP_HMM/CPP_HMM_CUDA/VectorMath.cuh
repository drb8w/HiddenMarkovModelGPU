#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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


