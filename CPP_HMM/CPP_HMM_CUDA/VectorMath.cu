#include "VectorMath.cuh"

#include "MemoryManagement.cuh"

__device__ unsigned int dev_glob_blocksize = 512; // value usually chosen by tuning and hardware constraints


__host__ unsigned int Min(unsigned int a, unsigned int b)
{
	return (a) < (b) ? a : b;
}

__host__ unsigned int Max(unsigned int a, unsigned int b)
{
	return (a) < (b) ? a : b;
}

//---------------------------------------------------------------------------------------------------------
// CPU - serial implementation
//---------------------------------------------------------------------------------------------------------

__host__ double sumElementMulMatrixHost(const double *host_U, const double *host_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
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

	for (unsigned int k = 0; k < dim1_U; k++)
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

__host__ double sumElementMulMatrixDevice(const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
	cudaError_t cudaStatus;
	double result = 0;

	//---------------------------------------------------------------------------------------------------------
	// memory allocation
	//---------------------------------------------------------------------------------------------------------
	double *dev_w = nullptr;

	cudaStatus = cudaMalloc((void**)dev_w, dim1_U * sizeof(double));

	//---------------------------------------------------------------------------------------------------------
	// actual calculation
	//---------------------------------------------------------------------------------------------------------

	// Launch a kernel on the GPU with one thread for several elements.
	// 1D
	//elementMulMatrixDevice << <1, blockDim.x >> >(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);
	unsigned int dimBlock = Min(dim1_U, dev_glob_blocksize);
	unsigned int dimGrid = ceil(dim1_U / (float)dimBlock);
	elementMulMatrixKernel <<<dimGrid, dimBlock >>>(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);

	//---------------------------------------------------------------------------------------------------------
	// syncronize
	//---------------------------------------------------------------------------------------------------------
	cudaStatus = cudaDeviceSynchronize();

	//---------------------------------------------------------------------------------------------------------
	// do sumation reduction
	//---------------------------------------------------------------------------------------------------------

	result = sumVectorDevice(dev_w, dim1_U, true);

	//---------------------------------------------------------------------------------------------------------
	// memory deallocation
	//---------------------------------------------------------------------------------------------------------

	cudaStatus = cudaFree(dev_w);

	return result;
}


__host__ double sumVectorDevice(double *dev_w, unsigned int dim_w, bool destructiveSummation)
{
	cudaError_t cudaStatus;
	double result = 0;

	bool parallelization = dim_w > dev_glob_blocksize;

	//---------------------------------------------------------------------------------------------------------
	// memory allocation
	//---------------------------------------------------------------------------------------------------------

	double *dev_sum = nullptr;
	if (destructiveSummation || !parallelization)
		dev_sum = dev_w;
	else
		cudaStatus = cudaMalloc((void**)dev_sum, dev_glob_blocksize * sizeof(double));

	//---------------------------------------------------------------------------------------------------------
	// actual summation
	//---------------------------------------------------------------------------------------------------------
	unsigned int dimBlock = dim_w;
	if (parallelization) 
	{
		// Launch a kernel on the GPU with one thread for several elements.
		// 1D
		dimBlock = dev_glob_blocksize;
		unsigned int dimGrid = ceil(dim_w / (float)dimBlock);
		sumVectorKernel <<<dimGrid, dimBlock >>>(dev_sum, dev_w, dimBlock, dim_w);

		//---------------------------------------------------------------------------------------------------------
		// syncronize
		//---------------------------------------------------------------------------------------------------------
		cudaStatus = cudaDeviceSynchronize(); // does this block the whole device when called from another kernel ???
	}
	//---------------------------------------------------------------------------------------------------------
	// final reduction
	//---------------------------------------------------------------------------------------------------------

	// serial loop
	for (int idx_i = 0; idx_i < dimBlock; idx_i++)
		result += dev_sum[idx_i];

	//---------------------------------------------------------------------------------------------------------
	// memory deallocation
	//---------------------------------------------------------------------------------------------------------
	if (!destructiveSummation)
		cudaStatus = cudaFree(dev_sum);


	return result;
}

__global__ void sumVectorKernel(double *dev_sum, const double *dev_w, unsigned int dim_sum, unsigned int dim_w)
{
	//---------------------------------------------------------------------------------------------------------
	// 1-level 'reduction' from level of dim_w to level of max(1, dim_sum (or below)) 
	//---------------------------------------------------------------------------------------------------------


	//---------------------------------------------------------------------------------------------------------
	// determine index
	//---------------------------------------------------------------------------------------------------------

	unsigned int idx_k_i = blockDim.x + threadIdx.x;
	unsigned int stepSize = gridDim.x * blockDim.x;
	unsigned int idx_k_j = idx_k_i + stepSize;

	if (idx_k_j > dim_w-1)
		return;

	//---------------------------------------------------------------------------------------------------------

	while(stepSize > dim_sum)
	{
		dev_sum[idx_k_i] = dev_w[idx_k_i] + dev_w[idx_k_j];

		__syncthreads(); // how to synchronize between different warps ???

		stepSize /= 2;
		idx_k_j = idx_k_i + stepSize;
	}


	//---------------------------------------------------------------------------------------------------------

}

__host__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
	cudaError_t cudaStatus;

	//---------------------------------------------------------------------------------------------------------
	
	// Launch a kernel on the GPU with one thread for several elements.
	// 1D
	//elementMulMatrixKernel << <1, blockDim.x >> >(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);
	unsigned int dimBlock = dev_glob_blocksize;
	unsigned int dimGrid = ceil(dim1_U / (float)dev_glob_blocksize);
	elementMulMatrixKernel <<<dimGrid, dimBlock >>>(dev_w, dev_U, dev_V, index_row_i, index_column_j, dim1_U, dim1_V);

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
	if (idx_k > dim1_U-1)
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
