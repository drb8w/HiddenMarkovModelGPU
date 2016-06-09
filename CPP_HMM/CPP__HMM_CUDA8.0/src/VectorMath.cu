#include "VectorMath.cuh"

__device__ int dev_glob_blocksize = 512; // value usually chosen by tuning and hardware constraints


//---------------------------------------------------------------------------------------------------------
// CPU - serial implementation
//---------------------------------------------------------------------------------------------------------

__host__ double rowColumnMulMatrixHost(const double *host_U, const double *host_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index - row first
	//---------------------------------------------------------------------------------------------------------

	unsigned int idx_u_i0 = index_row_i * dim1_U;
	unsigned int idx_v_0j = index_column_j;

	//---------------------------------------------------------------------------------------------------------
	// iterate the arrays - row major
	//---------------------------------------------------------------------------------------------------------
	double result = 0;

	for (int k = 0; k < dim1_U; k++)
	{
		unsigned int idx_u_ik = idx_u_i0 + k;
		unsigned int idx_v_kj = idx_v_0j + k*dim1_V;

		result += host_U[idx_u_ik] * host_V[idx_v_kj];
	}

	return result;
}

__host__ void elementMulMatrixHost(double *host_w, const double *host_U, const double *host_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index - row first
	//---------------------------------------------------------------------------------------------------------

	unsigned int idx_u_i0 = index_row_i * dim1_U;
	unsigned int idx_v_0j = index_column_j;

	//---------------------------------------------------------------------------------------------------------
	// iterate the arrays - row major
	//---------------------------------------------------------------------------------------------------------
	
	for (unsigned int idx_k = 0; idx_k < dim1_U; idx_k++)
	{
		unsigned int idx_u_ik = idx_u_i0 + idx_k;
		unsigned int idx_v_kj = idx_v_0j + idx_k*dim1_V;

		host_w[idx_k] = host_U[idx_u_ik] * host_V[idx_v_kj];
	}
}

//---------------------------------------------------------------------------------------------------------
// GPU - parallel implementation
// e.g. with reduction
//---------------------------------------------------------------------------------------------------------

__device__ double rowColumnMulMatrixDevice(const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
	cudaError_t cudaStatus;

	//---------------------------------------------------------------------------------------------------------
	// memory allocation
	//---------------------------------------------------------------------------------------------------------
	double *dev_w = nullptr;


	//---------------------------------------------------------------------------------------------------------
	// actual calculation
	//---------------------------------------------------------------------------------------------------------

	// Launch a kernel on the GPU with one thread for several elements.
	// 1D
	//elementMulMatrixDevice << <1, blockDim.x >> >(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);
	unsigned int dimBlock = dev_glob_blocksize;
	unsigned int dimGrid = ceil(dim1_U / (float)dev_glob_blocksize);
	elementMulMatrixKernel << <dimGrid, dimBlock >> >(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);

	//---------------------------------------------------------------------------------------------------------
	// syncronize
	//---------------------------------------------------------------------------------------------------------
	cudaStatus = cudaDeviceSynchronize();

	//---------------------------------------------------------------------------------------------------------
	// do sumation reduction
	//---------------------------------------------------------------------------------------------------------



	//---------------------------------------------------------------------------------------------------------
	// memory deallocation
	//---------------------------------------------------------------------------------------------------------



	return 0;
}

__global__ void sumReductionVectorKernel(double *dev_sum, const double *dev_w, unsigned int dim_w)
{

	//---------------------------------------------------------------------------------------------------------
	// determine index
	//---------------------------------------------------------------------------------------------------------


	//---------------------------------------------------------------------------------------------------------

}

__device__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
	cudaError_t cudaStatus;

	//---------------------------------------------------------------------------------------------------------
	
	// Launch a kernel on the GPU with one thread for several elements.
	// 1D
	//elementMulMatrixKernel << <1, blockDim.x >> >(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);
	unsigned int dimBlock = dev_glob_blocksize;
	unsigned int dimGrid = ceil(dim1_U / (float)dev_glob_blocksize);
	elementMulMatrixKernel << <dimGrid, dimBlock >> >(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);

	//---------------------------------------------------------------------------------------------------------
	// syncronize
	//---------------------------------------------------------------------------------------------------------
	cudaStatus = cudaDeviceSynchronize();

}

__global__ void elementMulMatrixKernel(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine indices - row first
	//---------------------------------------------------------------------------------------------------------

	// 1D
	//int idx_k = threadIdx.x;
	unsigned int idx_k = blockIdx.x * gridDim.x + threadIdx.x;

	// check index range to abort
	if (idx_k > dim1_U)
		return;

	unsigned int idx_u_i0 = index_row_i * dim1_U;
	unsigned int idx_v_0j = index_column_j;

	unsigned int idx_u_ik = idx_u_i0 + idx_k;
	unsigned int idx_v_kj = idx_v_0j + idx_k*dim1_V;

	//---------------------------------------------------------------------------------------------------------

	do
	{
		//---------------------------------------------------------------------------------------------------------
		// access the arrays - row major
		//---------------------------------------------------------------------------------------------------------

		dev_w[idx_k] = dev_U[idx_u_ik] * dev_V[idx_v_kj];

		//---------------------------------------------------------------------------------------------------------
		// determine new indices - row first
		//---------------------------------------------------------------------------------------------------------

		// 1D
		//idx_k += blockIdx.x;
		idx_k += blockIdx.x * gridDim.x;

		idx_u_ik = idx_u_i0 + idx_k;
		idx_v_kj = idx_v_0j + idx_k*dim1_V;

	} while (idx_k < dim1_U);
	//---------------------------------------------------------------------------------------------------------

}
