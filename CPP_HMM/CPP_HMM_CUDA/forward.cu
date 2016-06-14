#include "forward.cuh"

#include "MemoryManagement.cuh"

#include "Matricies.h"
#include "Observation.h"

#include "Utilities.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;


// ------------------------------------------------------------------------------------------------------
// global states
// ------------------------------------------------------------------------------------------------------
extern ComputationEnvironment glob_Env;
extern ComputationEnvironment trellis_3D_Env;

__host__ __device__ void createForwardIndices(int &idx_a_ji, int &idx_b_it, int &idx_p, int &idx_alpha_ti, int &idx_alpha_tm1j,  int i, int j, int t, int dim1_Alpha, int dim1_P, int dim2_P, int dim1_A, int dim1_B)
{
#ifdef	COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// COLUMN-MAJOR ORDER MATRIX: the first dimension in the array iterates the rows in the same column
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	// calculate transition and emmision index in 2D transition and emmision arrays of size dim1 * dim2:
	// a_ji
	idx_a_ji = j + i*dim1_A;
	// b_it
	idx_b_it = i + t*dim1_B;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = j + i*dim1_P + t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(t-1)j * p
	idx_alpha_ti = t + i*dim1_Alpha;
	idx_alpha_tm1j = (t - 1) + j*dim1_Alpha;
	// ------------------------------------------------------------------------------------------------------
#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// ROW-MAJOR ORDER MATRIX: the first dimension in the array iterates the columns in the same row
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	// calculate transition and emmision index in 2D transition and emmision arrays of size dim1 * dim2:
	// a_ji
	idx_a_ji = j*dim1_A + i;
	// b_it
	idx_b_it = i*dim1_B + t;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = j*dim1_P + i + t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(t-1)j * p
	idx_alpha_ti = t*dim1_Alpha + i;
	idx_alpha_tm1j = (t - 1)*dim1_Alpha + j;

#endif

}

__host__ __device__ void createForwardIndices2D(int &idx_a_ji, int &idx_b_it, int &idx_p, int i, int j, int t, int dim1_P, int dim2_P, int dim1_A, int dim1_B)
{
#ifdef	COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// COLUMN-MAJOR ORDER MATRIX: the first dimension in the array iterates the rows in the same column
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	// calculate transition and emmision index in 2D transition and emmision arrays of size dim1 * dim2:
	// a_ji
	idx_a_ji = j + i*dim1_A;
	// b_it
	idx_b_it = i + t*dim1_B;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = j + i*dim1_P + t*dim1_P*dim2_P;
	// ------------------------------------------------------------------------------------------------------
#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// ROW-MAJOR ORDER MATRIX: the first dimension in the array iterates the columns in the same row
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	// calculate transition and emmision index in 2D transition and emmision arrays of size dim1 * dim2:
	// a_ji
	idx_a_ji = j*dim1_A + i;
	// b_it
	idx_b_it = i*dim1_B + t;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = j*dim1_P + i + t*dim1_P*dim2_P;
	// ------------------------------------------------------------------------------------------------------
#endif

}

// ------------------------------------------------------------------------------------------------------

__host__ __device__ void forward(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, int i, int j, int t, int dim1_Alpha, int dim1_P, int dim2_P, int dim1_A, int dim1_B)
{
//#ifdef __CUDA_ARCH__
//	printf("Device Thread %d\n", threadIdx.x);
//#else
//	printf("Host code!\n");
//#endif

	// ------------------------------------------------------------------------------------------------------

	int idx_a_ji = 0;
	int idx_b_it = 0;
	int idx_p = 0;
	int idx_alpha_ti = 0;
	int idx_alpha_tm1j = 0;

	createForwardIndices(idx_a_ji, idx_b_it, idx_p, idx_alpha_ti, idx_alpha_tm1j, i, j, t, dim1_Alpha, dim1_P, dim2_P, dim1_A, dim1_B);

	// ------------------------------------------------------------------------------------------------------

	double a_ji = dev_A_stateTransProbs_2D[idx_a_ji];
	double b_it = dev_B_obsEmissionProbs_2D[idx_b_it];
	double p = a_ji * b_it;
	dev_probs_3D[idx_p] = p;
	dev_Alpha_trelis_2D[idx_alpha_ti] = dev_Alpha_trelis_2D[idx_alpha_ti] + dev_Alpha_trelis_2D[idx_alpha_tm1j] * p;
}

__host__ __device__ void forward2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, int i, int j, int t, int dim1_P, int dim2_P, int dim1_A, int dim1_B)
{
	//#ifdef __CUDA_ARCH__
	//	printf("Device Thread %d\n", threadIdx.x);
	//#else
	//	printf("Host code!\n");
	//#endif

	// ------------------------------------------------------------------------------------------------------

	int idx_a_ji = 0;
	int idx_b_it = 0;
	int idx_p = 0;

	createForwardIndices2D(idx_a_ji, idx_b_it, idx_p, i, j, t, dim1_P, dim2_P, dim1_A, dim1_B);
	// ------------------------------------------------------------------------------------------------------

	double a_ji = dev_A_stateTransProbs_2D[idx_a_ji];
	double b_it = dev_B_obsEmissionProbs_2D[idx_b_it];
	double p = a_ji * b_it;
	dev_probs_3D[idx_p] = p;
}

// ------------------------------------------------------------------------------------------------------

__device__ void createForwardMatrixDimensionsDevice(int &dim1_A, int &dim1_B, int &dim1_Alpha, int &dim1_P, int &dim2_P, int T_noOfObservations, int V_noOfObsSymbols)
{
#ifdef	COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// COLUMN-MAJOR ORDER MATRIX: the first dimension in the array iterates the rows in the same column
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	dim1_A = blockDim.x;
	//int dim2_A = gridDim.x; // would be number of states (in the column) but not needed here

	dim1_B = blockDim.x;
	//int dim2_B = V_noOfObsSymbols; // would be number of observation symbols but not needed here

	dim1_Alpha = T_noOfObservations; // size of observation sequence
	//int dim2_Alpha = blockDim.x;  // would be number of states (in the column) but not needed here

	dim1_P = blockDim.x;
	dim2_P = gridDim.x;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// ROW-MAJOR ORDER MATRIX: the first dimension in the array iterates the columns in the same row
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	dim1_A = gridDim.x; // number of states (in the row)
	//int dim2_A = blockDim.x; // would be number of states (in the column) but not needed here

	dim1_B = V_noOfObsSymbols; // number of observation symbols
	//int dim2_B =  blockDim.x; // would be number of states (in the column) but not needed here

	dim1_Alpha = blockDim.x; // number of states (in the row)
	//int dim2_Alpha = T_noOfObservations;  // would be size of observation sequence (in the column) but not needed here

	dim1_P = blockDim.x;
	dim2_P = gridDim.x;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

#endif
}

__device__ void createForwardMatrixDimensions2DDevice(int &dim1_A, int &dim1_B, int &dim1_P, int &dim2_P, int T_noOfObservations, int V_noOfObsSymbols)
{
//#ifdef	COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// COLUMN-MAJOR ORDER MATRIX: the first dimension in the array iterates the rows in the same column
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	dim1_A = blockDim.x;
	//int dim2_A = gridDim.x; // would be number of states (in the column) but not needed here

	dim1_B = blockDim.x;
	//int dim2_B = V_noOfObsSymbols; // would be number of observation symbols but not needed here

	dim1_P = blockDim.x;
	dim2_P = gridDim.x;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

//#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// ROW-MAJOR ORDER MATRIX: the first dimension in the array iterates the columns in the same row
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)


	dim1_A = gridDim.x; // number of states (in the row)
	//int dim2_A = blockDim.x; // would be number of states (in the column) but not needed here

	dim1_B = V_noOfObsSymbols; // number of observation symbols
	//int dim2_B =  blockDim.x; // would be number of states (in the column) but not needed here

	dim1_P = blockDim.x;
	dim2_P = gridDim.x;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

#endif
}

// ------------------------------------------------------------------------------------------------------

__global__ void forwardKernel(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *dev_O_obsSequence_1D, int T_noOfObservations, int idx_obs, int V_noOfObsSymbols)
{
	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	int dim1_A = 0;
	int dim1_B = 0;
	int dim1_Alpha = 0;
	int dim1_P = 0;
	int dim2_P = 0;
	
	createForwardMatrixDimensionsDevice(dim1_A, dim1_B, dim1_Alpha, dim1_P, dim2_P, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// determine indices
	// ------------------------------------------------------------------------------------------------------

	int i = blockIdx.x;
	int j = threadIdx.x;
	int t = idx_obs;

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	forward(dev_Alpha_trelis_2D, dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, i, j, t, dim1_Alpha, dim1_P, dim2_P, dim1_A, dim1_B);

}

__global__ void forwardKernel2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, int T_noOfObservations, int idx_obs, int V_noOfObsSymbols)
{
	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	int dim1_A = 0;
	int dim1_B = 0;
	int dim1_P = 0;
	int dim2_P = 0;

	createForwardMatrixDimensions2DDevice(dim1_A, dim1_B, dim1_P, dim2_P, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// determine indices
	// ------------------------------------------------------------------------------------------------------

	int i = blockIdx.x;
	int j = threadIdx.x;
	int t = idx_obs;

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	forward2D(dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, i, j, t, dim1_P, dim2_P, dim1_A, dim1_B);

}

// Serial dummy implementation - not to be used in final code
__host__ void ComputeBHost(const int M_noOfObsSequences, const int V_noOfObsSymbols, const int T_noOfObservations, const int N_noOfStates, const unsigned int *host_O_obsSequences_2D, const double *host_B_obsEmissionProbs_2D, const int j, double *B){

	for (int i = 0; i < M_noOfObsSequences; i++)
	{
		int idx_m_ij = i*T_noOfObservations + j;
		unsigned int value = host_O_obsSequences_2D[idx_m_ij];

		for (int n = 0; n < N_noOfStates; n++)
		{
			int idx = value*V_noOfObsSymbols + n;
			B[i*N_noOfStates + n] = host_B_obsEmissionProbs_2D[idx];
		}

	}

}

__global__ void ComputeBDevice(const int M_noOfObsSequences, const int V_noOfObsSymbols, const int T_noOfObservations, const int N_noOfStates, const unsigned int *dev_O_obsSequences_2D, const double *dev_B_obsEmissionProbs_2D, const int j, double *dev_B){
	
	// index to access element of B matrix
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// index to access device_Obs_seq
	int idx_m_ij = blockIdx.x *T_noOfObservations + j;
	unsigned int value = dev_O_obsSequences_2D[idx_m_ij];

	int emission_idx = value * V_noOfObsSymbols + threadIdx.x;
	dev_B[idx] = dev_B_obsEmissionProbs_2D[emission_idx];

}

// ------------------------------------------------------------------------------------------------------
// wrapper functions to switch transparently between GPU and CPU calcuation 
// without changing the main algorithms
// Call for the 3D Variant
// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ForwardAlgorithm(const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *dev_O_obsSequence_2D, int M_noOfObsSequences, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, unsigned int i, double * host_B, double * dev_3D_Trellis)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// choose environment of calculation
	// ------------------------------------------------------------------------------------------------------

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = ForwardAlgorithmGPU(dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_2D, M_noOfObsSequences, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, i, host_B, dev_3D_Trellis);
		break;
	case ComputationEnvironment::CPU:
		cudaStatus = ForwardAlgorithmCPU(dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, i, host_B, dev_3D_Trellis);
		break;
	}

	if (cudaStatus != cudaError_t::cudaSuccess)
		return cudaStatus;

	// ------------------------------------------------------------------------------------------------------
	// extract likelihood as the goal of the algorithm
	// likelihood = alpha_(Obs_T)endstate
	// ------------------------------------------------------------------------------------------------------

	//cudaStatus = CalculateLikelihoodAlphaTrellis2DHost(host_likelihood, host_Alpha_trelis_2D, N_noOfStates, T_noOfObservations);

	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithm2D(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// choose environment of calculation
	// ------------------------------------------------------------------------------------------------------

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = ForwardAlgorithm2DGPU(dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_probs_3D, host_likelihood);
		break;
	case ComputationEnvironment::CPU:
		cudaStatus = ForwardAlgorithm2DCPU(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_probs_3D, host_likelihood);
		break;
	}

	if (cudaStatus != cudaError_t::cudaSuccess)
		return cudaStatus;

	// ------------------------------------------------------------------------------------------------------
	// calculate AlphaTrellis2D in a serial fashion
	// ------------------------------------------------------------------------------------------------------

	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	int dim1_A = 0;
	int dim1_B = 0;
	int dim1_Alpha = 0;
	int dim1_P = 0;
	int dim2_P = 0;

	createForwardMatrixDimensionsHost(dim1_A, dim1_B, dim1_Alpha, dim1_P, dim2_P, N_noOfStates, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------

	for (int t = 1; t < T_noOfObservations; t++)
	{
		for (int i = 0; i < N_noOfStates; i++)
		{
			for (int j = 0; j < N_noOfStates; j++)
			{
				// ------------------------------------------------------------------------------------------------------
				// determine indices
				// ------------------------------------------------------------------------------------------------------

				int idx_a_ji = 0;
				int idx_b_it = 0;
				int idx_p = 0;
				int idx_alpha_ti = 0;
				int idx_alpha_tm1j = 0;

				createForwardIndices(idx_a_ji, idx_b_it, idx_p, idx_alpha_ti, idx_alpha_tm1j, i, j, t, dim1_Alpha, dim1_P, dim2_P, dim1_A, dim1_B);

				// ------------------------------------------------------------------------------------------------------
				// actual calculation
				// ------------------------------------------------------------------------------------------------------
				double p = host_probs_3D[idx_p];
				double v = host_Alpha_trelis_2D[idx_alpha_ti] + host_Alpha_trelis_2D[idx_alpha_tm1j] * p;
				host_Alpha_trelis_2D[idx_alpha_ti] = v;

			}
		}
	}

	// ------------------------------------------------------------------------------------------------------
	// extract likelihood as the goal of the algorithm
	// likelihood = alpha_(Obs_T)endstate
	// ------------------------------------------------------------------------------------------------------

	cudaStatus = CalculateLikelihoodAlphaTrellis2DHost(host_likelihood, host_Alpha_trelis_2D, N_noOfStates, T_noOfObservations);

	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithmGPU(const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *dev_O_obsSequence_2D, int M_noOfObsSequences,int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, unsigned int i, double * host_B, double * dev_3D_Trellis)
{
	cudaError_t cudaStatus;
	double *dev_probs_3D = nullptr;
	double* dev_B = nullptr;
	double* dev_W = nullptr;
	double* dev_D = nullptr;

	// --------------------------------------------------------------------------------------------------------
	// device memory allocation
	// --------------------------------------------------------------------------------------------------------


	if ((cudaStatus = allocateDeviceVector(&dev_B, M_noOfObsSequences * N_noOfStates)) != cudaSuccess) {
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_W, M_noOfObsSequences * N_noOfStates)) != cudaSuccess) {
		deviceFree(dev_B);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_D, M_noOfObsSequences * M_noOfObsSequences)) != cudaSuccess) {
		deviceFree(dev_B);
		deviceFree(dev_W);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// copy memory from host to device
	// ------------------------------------------------------------------------------------------------------
	

	//if ((cudaStatus = memcpyVector(dev_B, host_B, T_noOfObservations * N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
	//	deviceFree(dev_B);
	//	return cudaStatus;
	//}

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	// Computes the B matrix in term D = B .* C x A
	ComputeBDevice << < M_noOfObsSequences, N_noOfStates >> >(M_noOfObsSequences, V_noOfObsSymbols, T_noOfObservations, N_noOfStates, dev_O_obsSequence_2D, dev_B_obsEmissionProbs_2D, i, dev_B);

	cudaStatus = cudaDeviceSynchronize();

	// All dimensions are multipe of Warp Sizes
	// Compute W =  B .* C
	poitwiseMatrixMul << <M_noOfObsSequences, N_noOfStates >> >(dev_W, dev_B, &dev_3D_Trellis[(i - 1) * M_noOfObsSequences * N_noOfStates]);

	cudaStatus = cudaDeviceSynchronize();

	// Compute D = W x A
	cublasMultiplyDouble(M_noOfObsSequences, N_noOfStates, N_noOfStates, dev_W, dev_A_stateTransProbs_2D, &dev_3D_Trellis[i * M_noOfObsSequences * N_noOfStates]);
	

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	// ------------------------------------------------------------------------------------------------------


	// ------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// ------------------------------------------------------------------------------------------------------
	deviceFree(dev_B);
	deviceFree(dev_W);
	deviceFree(dev_D);

	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithm2DGPU(const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_probs_3D, double &host_likelihood)
{
	cudaError_t cudaStatus;
	double *dev_probs_3D = nullptr;
	unsigned int *dev_O_obsSequence_1D = nullptr;

	// --------------------------------------------------------------------------------------------------------
	// device memory allocation
	// --------------------------------------------------------------------------------------------------------

	// array to store the observation sequence
	if ((cudaStatus = allocateDeviceVector(&dev_O_obsSequence_1D, T_noOfObservations)) != cudaSuccess) {
		return cudaStatus;
	}

	// array to store all probabilities.
	if ((cudaStatus = allocateDeviceVector(&dev_probs_3D, N_noOfStates * N_noOfStates * T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_O_obsSequence_1D);
		return cudaStatus;
	}


	// ------------------------------------------------------------------------------------------------------
	// copy memory from host to device
	// ------------------------------------------------------------------------------------------------------


	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	for (unsigned int idx_obs = 1; idx_obs < T_noOfObservations; idx_obs++){

		// call kernel for NxT matrix ops (N is the number of states, T is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.
		forwardKernel2D << <N_noOfStates, N_noOfStates >> >(dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, T_noOfObservations, idx_obs, V_noOfObsSymbols);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	// ------------------------------------------------------------------------------------------------------

	if (cudaStatus != cudaSuccess) {
		deviceFree(dev_O_obsSequence_1D);
		deviceFree(dev_probs_3D);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// copy memory from device to host
	// ------------------------------------------------------------------------------------------------------

	// Copy output vector from GPU buffer to host memory.
	if ((cudaStatus = memcpyVector(host_probs_3D, dev_probs_3D, N_noOfStates * N_noOfStates * T_noOfObservations, cudaMemcpyDeviceToHost)) != cudaSuccess) {
		deviceFree(dev_O_obsSequence_1D);
		deviceFree(dev_probs_3D);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// ------------------------------------------------------------------------------------------------------
	deviceFree(dev_probs_3D);

	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithmCPU(const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, unsigned int i, double * host_B, double * dev_3D_Trellis)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	int dim1_A = 0;
	int dim1_B = 0;
	int dim1_Alpha = 0;
	int dim1_P = 0;
	int dim2_P = 0;

	createForwardMatrixDimensionsHost(dim1_A, dim1_B, dim1_Alpha, dim1_P, dim2_P, N_noOfStates, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// determine indices
	// ------------------------------------------------------------------------------------------------------

	// t = idx_obs
	for (unsigned int t = 1; t < T_noOfObservations; t++){

		// call kernel for NxT matrix ops (N is the number of states, T is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.
		
		//forwardKernel << <N_noOfStates, N_noOfStates >> >(dev_Alpha_trelis_2D, dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, T_noOfObservations, idx_obs, V_noOfObsSymbols);
		for (int i = 0; i < N_noOfStates; i++)
		{
			for (int j = 0; j < N_noOfStates; j++)
			{
				// ------------------------------------------------------------------------------------------------------
				// actual calculation
				// ------------------------------------------------------------------------------------------------------

				// make kernel callable as normal function on host
				//forward(host_Alpha_trelis_2D, host_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, i, j, t, dim1_Alpha, dim1_P, dim2_P, dim1_A, dim1_B);
			}
		}
	}

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithm2DCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	int dim1_A = 0;
	int dim1_B = 0;
	int dim1_Alpha = 0;
	int dim1_P = 0;
	int dim2_P = 0;

	createForwardMatrixDimensionsHost(dim1_A, dim1_B, dim1_Alpha, dim1_P, dim2_P, N_noOfStates, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// determine indices
	// ------------------------------------------------------------------------------------------------------

	// t = idx_obs
	for (unsigned int t = 1; t < T_noOfObservations; t++){

		// call kernel for NxT matrix ops (N is the number of states, T is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.

		//forwardKernel << <N_noOfStates, N_noOfStates >> >(dev_Alpha_trelis_2D, dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, T_noOfObservations, idx_obs, V_noOfObsSymbols);
		for (int i = 0; i < N_noOfStates; i++)
		{
			for (int j = 0; j < N_noOfStates; j++)
			{
				// ------------------------------------------------------------------------------------------------------
				// actual calculation
				// ------------------------------------------------------------------------------------------------------

				// make kernel callable as normal function on host
				forward2D(host_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, i, j, t, dim1_P, dim2_P, dim1_A, dim1_B);

			}
		}
	}

	return cudaStatus;
}

// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t CalculateLikelihoodAlphaTrellis2DHost(double &host_likelihood, const double *host_Alpha_trelis_2D, int N_noOfStates, int T_noOfObservations)
{
	cudaError_t cudaStatus = cudaError_t::cudaSuccess;

	// ------------------------------------------------------------------------------------------------------
	// extract likelihood as the goal of the algorithm
	// likelihood = alpha_(Obs_T)endstate
	// ------------------------------------------------------------------------------------------------------

	//// get index of last obervation symbol in set of observation symbols
	//int idx_obs_T = 0;
	//// TODO: similar to the following
	////Observation observation;
	////idx_obs_T = observation.getObservationSymbolIndex(dev_O_obsSequence_1D[T_noOfObservations-1]);
	//// HACK: symbol id is same as index
	//int obs_T = host_O_obsSequence_1D[T_noOfObservations - 1];
	//idx_obs_T =	obs_T;
	//
	//// get index of end state in set of states
	//int idx_state_end = 0;
	//// TODO: similar to the following
	////Matricies mat;
	////int idx_state_end = mat.getStateIndex(state_end);
	//// get index in trellis and return as likelihood
	//int idx_alpha_obsT_stateEnd = idx_obs_T + idx_state_end *T_noOfObservations;
	//host_likelihood = host_Alpha_trelis_2D[idx_alpha_obsT_stateEnd];

	// ------------------------------------------------------------------------------------------------------
	// likelihood as sum of last row in trellis
	// TODO: in GPU fashion this can be done with a reduction, but maybe not nec.

	host_likelihood = 0;

#ifdef COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	int dim1_Alpha = T_noOfObservations;
	int dim2_Alpha = N_noOfStates;
	for (int i = 0; i < dim1_Alpha; i++) {
		int idx_alpha_Ti = (T_noOfObservations - 1) + i*dim1_Alpha;
		host_likelihood += host_Alpha_trelis_2D[idx_alpha_Ti];
	}
#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
	int dim1_Alpha = N_noOfStates;
	int dim2_Alpha = T_noOfObservations;
	for (int i = 0; i < dim1_Alpha; i++) {
		int idx_alpha_Ti = (T_noOfObservations - 1)*dim1_Alpha + i;
		host_likelihood += host_Alpha_trelis_2D[idx_alpha_Ti];
	}
#endif

	return cudaStatus;
}

__host__ void createForwardMatrixDimensionsHost(int &dim1_A, int &dim1_B, int &dim1_Alpha, int &dim1_P, int &dim2_P, int N_noOfStates, int T_noOfObservations, int V_noOfObsSymbols)
{
#ifdef	COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// COLUMN-MAJOR ORDER MATRIX: the first dimension in the array iterates the rows in the same column
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	dim1_A = N_noOfStates; // number of states (in the row) but not needed here
	//int dim2_A = N_noOfStates; // would be number of states (in the column) but not needed here

	dim1_B = N_noOfStates; // number of states(in the row) but not needed here
	//int dim2_B = V_noOfObsSymbols; // would be number of observation symbols but not needed here

	dim1_Alpha = T_noOfObservations; // size of observation sequence
	//int dim2_Alpha = blockDim.x;  // would be number of states (in the column) but not needed here

	dim1_P = N_noOfStates;
	dim2_P = N_noOfStates;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// ROW-MAJOR ORDER MATRIX: the first dimension in the array iterates the columns in the same row
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	dim1_A = N_noOfStates; // number of states (in the row)
	//int dim2_A = N_noOfStates; // would be number of states (in the column) but not needed here

	dim1_B = V_noOfObsSymbols; // number of observation symbols
	//int dim2_B =  N_noOfStates; // would be number of states (in the column) but not needed here

	dim1_Alpha = N_noOfStates; // number of states (in the row)
	//int dim2_Alpha = T_noOfObservations;  // would be size of observation sequence (in the column) but not needed here

	dim1_P = N_noOfStates;
	dim2_P = N_noOfStates;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

	// ------------------------------------------------------------------------------------------------------
#endif
}

// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ForwardAlgorithmSet(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, int M_noOfObsSequences, double *host_likelihoods_1D)
{
	cout << "starting 3D fw alg for obs sequence...\n";

	double *dev_Pi_startProbs_1D = nullptr;
	double *dev_A_stateTransProbs_2D = nullptr;
	double *dev_B_obsEmissionProbs_2D = nullptr;
	unsigned int *dev_O_obsSequences_2D = nullptr;
	double *dev_3D_Trellis = nullptr;
	cudaError_t cudaStatus = cudaSuccess;

	// --------------------------------------------------------------------------------------------------------
	// device memory allocation
	// --------------------------------------------------------------------------------------------------------

	if ((cudaStatus = allocateDeviceVector(&dev_Pi_startProbs_1D, N_noOfStates)) != cudaSuccess) {
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_A_stateTransProbs_2D, N_noOfStates*N_noOfStates)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_3D_Trellis, M_noOfObsSequences * N_noOfStates*T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_O_obsSequences_2D,T_noOfObservations* M_noOfObsSequences)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_3D_Trellis);
		return cudaStatus;
	}

	

	// --------------------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------------------
	// memory copy from host do device
	// --------------------------------------------------------------------------------------------------------
	// Copy input vectors from host memory to GPU buffers.
	if ((cudaStatus = memcpyVector(dev_Pi_startProbs_1D, (double *)host_Pi_startProbs_1D, N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_A_stateTransProbs_2D, (double *)host_A_stateTransProbs_2D, N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_B_obsEmissionProbs_2D, (double *)host_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_O_obsSequences_2D, (unsigned int *)host_O_obsSequences_2D, T_noOfObservations* M_noOfObsSequences, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	double* host_Init_Alpha_trelis_2D = (double *)calloc(M_noOfObsSequences * N_noOfStates, sizeof(double));
	unsigned int* host_O_obsSequence_1D = nullptr;
	
	switch (trellis_3D_Env)
	{
	case ComputationEnvironment::CPU:

		// for each obs. sequence do : Init first C layer of 3D trellis
		for (unsigned int i = 0; i < M_noOfObsSequences; i++) {

			// extract the right pointer position out of host_O_obsSequences_2D
			int dim1_M = T_noOfObservations;
			host_O_obsSequence_1D = (unsigned int *)(host_O_obsSequences_2D + (i*dim1_M)); // seems to be ok

			// --------------------------------------------------------------------------------------------------------
			// host memory initialization
			// --------------------------------------------------------------------------------------------------------

			AlphaTrellisInitialization2D(host_Init_Alpha_trelis_2D, host_Pi_startProbs_1D, host_B_obsEmissionProbs_2D, host_O_obsSequence_1D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols, M_noOfObsSequences, i);

		}

		// Copy initial C slice into first layer of 3D trellis
		if ((cudaStatus = memcpyVector(dev_3D_Trellis, host_Init_Alpha_trelis_2D, M_noOfObsSequences * N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_3D_Trellis);
			deviceFree(dev_O_obsSequences_2D);
			return cudaStatus;
		}

		free(host_Init_Alpha_trelis_2D);

		break;
	case ComputationEnvironment::GPU:
		AlphaTrellisInitializationGPU << <M_noOfObsSequences, N_noOfStates >> >(dev_3D_Trellis, dev_Pi_startProbs_1D, dev_B_obsEmissionProbs_2D, dev_O_obsSequences_2D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		break;
	}

	// --------------------------------------------------------------------------------------------------------
	// actual calculation
	// Compute a 2D Slice of the 3D trellis: dim M x N
	// --------------------------------------------------------------------------------------------------------
	double * B = (double*)calloc(M_noOfObsSequences*N_noOfStates, sizeof(double));

	for (unsigned int i = 1; i<T_noOfObservations; i++){

		cudaError_t cudaStatus = ForwardAlgorithm(dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequences_2D, M_noOfObsSequences, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, i, B, dev_3D_Trellis);

		if (cudaStatus != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_3D_Trellis);
			deviceFree(dev_O_obsSequences_2D);
			return cudaStatus;
		}

	}

	double * host_3D_trellis = (double *)calloc(M_noOfObsSequences * N_noOfStates, sizeof(double));

	// Copy the last slice of the 3D trellis form device onto host
	if ((cudaStatus = memcpyVector(host_3D_trellis, &dev_3D_Trellis[(T_noOfObservations - 1) * M_noOfObsSequences * N_noOfStates], M_noOfObsSequences * N_noOfStates, cudaMemcpyDeviceToHost)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------
	// extract likelihood
	// --------------------------------------------------------------------------------------------------------
	// fill host_likelihoods_1D

	for (int i = 0; i < M_noOfObsSequences; i++)
	{
		for (int j = 0; j < N_noOfStates; j++){
			host_likelihoods_1D[i] += host_3D_trellis[i*j];
		}

		cout << "likelihood: "  << host_likelihoods_1D[i] << "\n";
	}

	// --------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// --------------------------------------------------------------------------------------------------------
	deviceFree(dev_Pi_startProbs_1D);
	deviceFree(dev_A_stateTransProbs_2D);
	deviceFree(dev_B_obsEmissionProbs_2D);
	deviceFree(dev_O_obsSequences_2D);
	deviceFree(dev_3D_Trellis);

	// --------------------------------------------------------------------------------------------------------

#endif
}

__host__ cudaError_t ForwardAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, int M_noOfObsSequences, double *host_likelihoods_1D)
{
	double *dev_Pi_startProbs_1D = nullptr;
	double *dev_A_stateTransProbs_2D = nullptr;
	double *dev_B_obsEmissionProbs_2D = nullptr;
	cudaError_t cudaStatus = cudaSuccess;

	// --------------------------------------------------------------------------------------------------------
	// device memory allocation
	// --------------------------------------------------------------------------------------------------------

	if ((cudaStatus = allocateDeviceVector(&dev_Pi_startProbs_1D, N_noOfStates)) != cudaSuccess) {
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_A_stateTransProbs_2D, N_noOfStates*N_noOfStates)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------------------
	// memory copy from host do device
	// --------------------------------------------------------------------------------------------------------
	// Copy input vectors from host memory to GPU buffers.
	if ((cudaStatus = memcpyVector(dev_Pi_startProbs_1D, (double *)host_Pi_startProbs_1D, N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_A_stateTransProbs_2D, (double *)host_A_stateTransProbs_2D, N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_B_obsEmissionProbs_2D, (double *)host_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// TODO: the M different observation sequences could be computed in parallel

	// for each obs. sequence do
	for (unsigned int i = 0; i<M_noOfObsSequences; i++) {

		cout << "starting fw alg for obs sequence...\n";

		// --------------------------------------------------------------------------------------------------------
		// host memory allocation
		// --------------------------------------------------------------------------------------------------------
		double* host_Alpha_trelis_2D = (double *)calloc(T_noOfObservations * N_noOfStates, sizeof(double));
		double* host_probs_3D = (double *)calloc(N_noOfStates * N_noOfStates * T_noOfObservations, sizeof(double));

		// extract the right pointer position out of host_O_obsSequences_2D
		int dim1_M = T_noOfObservations;
		unsigned int* host_O_obsSequence_1D = nullptr;
		host_O_obsSequence_1D = (unsigned int *)(host_O_obsSequences_2D + (i*dim1_M)); // seems to be ok

		// --------------------------------------------------------------------------------------------------------
		// host memory initialization
		// --------------------------------------------------------------------------------------------------------

		AlphaTrellisInitialization1D(host_Alpha_trelis_2D, host_Pi_startProbs_1D, host_B_obsEmissionProbs_2D, host_O_obsSequence_1D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

		

		// --------------------------------------------------------------------------------------------------------
		// actual calculation
		// --------------------------------------------------------------------------------------------------------

		double host_likelihood = 0;
		cudaError_t cudaStatus = ForwardAlgorithm2D(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_probs_3D, host_likelihood);

		if (cudaStatus != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			return cudaStatus;
		}

		// --------------------------------------------------------------------------------------------------------
		// extract likelihood
		// --------------------------------------------------------------------------------------------------------
		// fill host_likelihoods_1D
		host_likelihoods_1D[i] = host_likelihood;

		cout << "likelihood :" << host_likelihood << "\n";

		// --------------------------------------------------------------------------------------------------------
		// host memory cleanup
		// --------------------------------------------------------------------------------------------------------
		free(host_Alpha_trelis_2D);
		free(host_probs_3D);
		// --------------------------------------------------------------------------------------------------------
	}

	// --------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// --------------------------------------------------------------------------------------------------------
	deviceFree(dev_Pi_startProbs_1D);
	deviceFree(dev_A_stateTransProbs_2D);
	deviceFree(dev_B_obsEmissionProbs_2D);

	// --------------------------------------------------------------------------------------------------------

#endif

}

// ------------------------------------------------------------------------------------------------------

__host__ void AlphaTrellisInitialization1D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols)
{

	// ------------------------------------------------------------------------------------------------------
	// Initialization of the Alpha_trelis
	// in the paper the initialization of the trellis is done differently, in code actually it is an initialization from the priors
	// ------------------------------------------------------------------------------------------------------

	// a_0i = pi_i --- actually data should be set up like this, but to be sure Pi is transported in an extra vector
	// alpha_0(i) = Pi_i*b_i(O_0)

	int obs_start = host_O_obsSequence_1D[0];
	// TODO: similar to the following
	//Observation observation;
	//idx_obs_T = observation.getObservationSymbolIndex(obs_start);
	// HACK: symbol id is same as index
	int idx_obs_start = obs_start;

#ifdef COL_MAJ_ORD_MAT_ROW_FIRST_INDEX

	int dim1_B = N_noOfStates;
	int dim1_Alpha = T_noOfObservations;
	// init first row of trellis
	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		int idx_b_i_idxOs = i + idx_obs_start * dim1_B;
		int idx_alpha_0i = i*dim1_Alpha;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// init first row of trellis
	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		int idx_b_i_idxOs = i*V_noOfObsSymbols + idx_obs_start;
		int idx_alpha_0i = i ;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif
}

__host__ void AlphaTrellisInitialization2D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols, int M_noOfSequences, int j)
{

	// ------------------------------------------------------------------------------------------------------
	// Initialization of the Alpha_trelis
	// in the paper the initialization of the trellis is done differently, in code actually it is an initialization from the priors
	// ------------------------------------------------------------------------------------------------------

	// a_0i = pi_i --- actually data should be set up like this, but to be sure Pi is transported in an extra vector
	// alpha_0(i) = Pi_i*b_i(O_0)

	int obs_start = host_O_obsSequence_1D[0];
	// TODO: similar to the following
	//Observation observation;
	//idx_obs_T = observation.getObservationSymbolIndex(obs_start);
	// HACK: symbol id is same as index
	int idx_obs_start = obs_start;

#ifdef COL_MAJ_ORD_MAT_ROW_FIRST_INDEX

	int dim1_B = N_noOfStates;
	int dim1_Alpha = T_noOfObservations;
	// init first row of trellis
	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		int idx_b_i_idxOs = i + idx_obs_start * dim1_B;
		int idx_alpha_0i = i*dim1_Alpha;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// init first row of trellis
	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		int idx_b_i_idxOs = i*V_noOfObsSymbols + idx_obs_start;
		int idx_alpha_0i = j*N_noOfStates+ i;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif
}

__global__ void AlphaTrellisInitializationGPU(double *dev_3D_Trellis, const double *dev_Pi_startProbs_1D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *dev_O_obsSequences_2D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols){

	int obs_index = blockIdx.x * T_noOfObservations;
	int obs_start = dev_O_obsSequences_2D[obs_index];
	int idx_b_i_idxOs = threadIdx.x*V_noOfObsSymbols + obs_start;
	int idx_alpha_0i = blockIdx.x * N_noOfStates + threadIdx.x;
	int idx_pi_i = threadIdx.x;

	double alpha_0_i = dev_Pi_startProbs_1D[idx_pi_i] * dev_B_obsEmissionProbs_2D[idx_b_i_idxOs];
	dev_3D_Trellis[idx_alpha_0i] = alpha_0_i;


}

