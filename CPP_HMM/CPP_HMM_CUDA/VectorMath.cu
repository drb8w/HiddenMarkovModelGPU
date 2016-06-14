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

#ifdef CUDA_35
__device__ double sumElementMulMatrixDevice(const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
#else
__host__ double sumElementMulMatrixDevice(const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
#endif
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

#ifdef CUDA_35
__device__ double sumVectorDevice(double *dev_w, unsigned int dim_w, bool destructiveSummation)
#else
__host__ double sumVectorDevice(double *dev_w, unsigned int dim_w, bool destructiveSummation)
#endif
{
	cudaError_t cudaStatus = cudaSuccess;
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
	cudaError_t cudaStatus = cudaSuccess;

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

#ifdef CUDA_35
__device__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
#else
__host__ void elementMulMatrixDevice(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
#endif
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

//---------------------------------------------------------------------------------------------------------

__host__ void calculateAlphaTrellis3DTimeslice(double *dev_Alpha3D, const double *dev_B, const double *dev_A, unsigned int M_noOfObsSequences, unsigned int N_noOfStates, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols)
{
#ifdef CUDA_35
	// call kernels for D_ij calculation in parallel
#else
	// call kernels for D_ij calculation with OpenMP to emulate CUDA 3.5 - slow because only maybe 8 CPU cores

	// alternatively: calculate everything till start of reduction
	// use CPU synchronization with every reduction step 

	// alternatively:
	// do simply everything serial in the kernel for D_ij as serial because the threads are running in parallel anyhow!

#endif
}

__host__ void cublasMultiplyDouble(int row_A, int col_B, int col_A, const double* A_dev,const double* B_dev, double* C_dev){

	const double alpha = 1.0;
	const double beta  = 0.0;
	cublasOperation_t matrix_orientation = CUBLAS_OP_N;
	cudaError_t cudaStatus = cudaSuccess;
	cublasHandle_t handle;
	cublasCreate(&handle);

	int m, n, k;
	const double* A_local;
	const double* B_local;


#ifdef COL_MAJ_ORD_MAT_ROW_FIRST_INDEX

	m = row_A;
	n = col_B;
	k = col_A;

	A_local = A_dev;
	B_local = B_dev;

#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
	
	// have to assign parameters with regard to B(T) * A(T)

	m = col_B; 
	n = row_A;
	k = col_A;

	A_local = B_dev;
	B_local = A_dev;

#endif

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_local, m, B_local, k, &beta, C_dev, m);
}

__global__ void pointwiseMatrixMul(double * dev_w, double *dev_A, double* dev_B){

	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int idx_k = ix;

	dev_w[idx_k] = dev_A[idx_k] * dev_B[idx_k];

}