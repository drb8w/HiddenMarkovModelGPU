#include "VectorMath.cuh"

//---------------------------------------------------------------------------------------------------------
// CPU - serial implementation
//---------------------------------------------------------------------------------------------------------

__host__ double rowColumnMulMatrixHost(const double *host_U, const double *host_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index - row first
	//---------------------------------------------------------------------------------------------------------

	int idx_u_i0 = index_row_i * dim1_U;
	int idx_v_0j = index_column_j;

	//---------------------------------------------------------------------------------------------------------
	// iterate the arrays - row major
	//---------------------------------------------------------------------------------------------------------
	double result = 0;

	for (int k = 0; k < dim1_U; k++)
	{
		int idx_u_ik = idx_u_i0 + k;
		int idx_v_kj = idx_v_0j + k*dim1_V;

		result += host_U[idx_u_ik] * host_V[idx_v_kj];
	}

	return result;
}

__host__ void elementMulMatrixHost(double *host_w, const double *host_U, const double *host_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index - row first
	//---------------------------------------------------------------------------------------------------------

	int idx_u_i0 = index_row_i * dim1_U;
	int idx_v_0j = index_column_j;

	//---------------------------------------------------------------------------------------------------------
	// iterate the arrays - row major
	//---------------------------------------------------------------------------------------------------------
	
	for (int idx_k = 0; idx_k < dim1_U; idx_k++)
	{
		int idx_u_ik = idx_u_i0 + idx_k;
		int idx_v_kj = idx_v_0j + idx_k*dim1_V;

		host_w[idx_k] = host_U[idx_u_ik] * host_V[idx_v_kj];
	}
}

//---------------------------------------------------------------------------------------------------------
// GPU - parallel implementation
// e.g. with reduction
//---------------------------------------------------------------------------------------------------------

__device__ double rowColumnMulMatrixDevice(const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index
	//---------------------------------------------------------------------------------------------------------

	// second dimension of matrices needed to stop additional threads from executing

	return 0;
}

__global__ void rowColumnMulMatrixKernel(double *dev_w, const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{

	//---------------------------------------------------------------------------------------------------------
	// determine index
	//---------------------------------------------------------------------------------------------------------


	//---------------------------------------------------------------------------------------------------------

}

__device__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{
	cudaError_t cudaStatus;

	// second dimension of matrices needed to stop additional threads from executing

	//---------------------------------------------------------------------------------------------------------


	// Launch a kernel on the GPU with one thread for each element.
	elementMulMatrixKernel << <1, dim1_U >> >(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);


	//---------------------------------------------------------------------------------------------------------
	// syncronize
	//---------------------------------------------------------------------------------------------------------
	cudaStatus = cudaDeviceSynchronize();

	//---------------------------------------------------------------------------------------------------------
	// do reduction
	//---------------------------------------------------------------------------------------------------------



}

__global__ void elementMulMatrixKernel(double *dev_w, const double *dev_U, const double *dev_V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index - row first
	//---------------------------------------------------------------------------------------------------------

	int idx_k = threadIdx.x;

	// check index range to abort
	if (idx_k > dim1_U)
		return;

	int idx_u_i0 = index_row_i * dim1_U;
	int idx_v_0j = index_column_j;

	int idx_u_ik = idx_u_i0 + idx_k;
	int idx_v_kj = idx_v_0j + idx_k*dim1_V;

	//---------------------------------------------------------------------------------------------------------
	// access the arrays - row major
	//---------------------------------------------------------------------------------------------------------

	dev_w[idx_k] = dev_U[idx_u_ik] * dev_V[idx_v_kj];
	
	//---------------------------------------------------------------------------------------------------------

}
