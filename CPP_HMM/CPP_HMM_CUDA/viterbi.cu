#include "viterbi.cuh"

#include "MemoryManagement.cuh"
#include "Trellis.cuh"

#include "Matricies.h"
#include "Observation.h"

#include "Utilities.h"
#include "VectorMath.cuh"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;


// ------------------------------------------------------------------------------------------------------
// global states
// ------------------------------------------------------------------------------------------------------
extern ComputationEnvironment glob_Env;
extern unsigned int glob_blocksize;

// ------------------------------------------------------------------------------------------------------

__host__ __device__ void viterbi1D(double *dev_Alpha_trelis_2D, unsigned int *dev_Gamma_trellis_backtrace_2D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_Alpha, unsigned int dim1_A, unsigned int dim1_B)
{
	
	// ------------------------------------------------------------------------------------------------------
	// determine indices
	// ------------------------------------------------------------------------------------------------------
	unsigned int idx_a_ji = 0;
	unsigned int idx_b_it = 0;
	unsigned int idx_alpha_ti = 0;
	unsigned int idx_alpha_tm1j = 0;

	createViterbiIndices1D(idx_a_ji, idx_b_it, idx_alpha_ti, idx_alpha_tm1j, idx_i, idx_j, idx_t, dim1_A, dim1_B, dim1_Alpha);

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	double a_ji = dev_A_stateTransProbs_2D[idx_a_ji];
	double b_it = dev_B_obsEmissionProbs_2D[idx_b_it];
	double p = a_ji * b_it;
	double alpha_tm1j = dev_Alpha_trelis_2D[idx_alpha_tm1j];
	double alpha_ti = dev_Alpha_trelis_2D[idx_alpha_ti];
	double partPathProb_tm1j = alpha_tm1j * p;
	if (partPathProb_tm1j >alpha_ti)
	{
		dev_Alpha_trelis_2D[idx_alpha_ti] = partPathProb_tm1j;
		// backpointers(t,i) = j
		dev_Gamma_trellis_backtrace_2D[idx_alpha_ti] = idx_j;
	}
}

__global__ void viterbiKernel1D(const double *dev_Pi_startProbs_1D, const double *dev_Alpha_trelis_TNM_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D,
	const unsigned int *dev_O_obsSequences_2D, unsigned int T_noOfObservations, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int M_noOfObsSequences,
	unsigned int *dev_Gamma_trellis_backtrace_TNM_3D, unsigned int *dev_likeliestStateIndexSequence_2D)
{
	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	unsigned int dim1_Alpha = 0;
	unsigned int dim2_Alpha = 0;
	unsigned int dim1_A = 0;
	unsigned int dim1_B = 0;

	createViterbiMatrixDimensions1D(dim1_Alpha, dim2_Alpha, dim1_A, dim1_B, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

	unsigned int dim1_Gamma = dim1_Alpha;
	unsigned int dim2_Gamma = dim2_Alpha;
	unsigned int dim1_obsSeq = dim2_Alpha;

	// ------------------------------------------------------------------------------------------------------
	// slice dev_Alpha_trelis_3D and dev_Gamma_trellis_backtrace_3D to 2D according to idx_m
	// slice dev_likeliestStateIndexSequence_1D to 1D according to idx_m
	// ------------------------------------------------------------------------------------------------------

	unsigned int idx_m = blockIdx.x * blockDim.x + threadIdx.x;

	// check if idx_m is out of range and abort this kernel thread
	if (idx_m > M_noOfObsSequences)
		return;

	double *dev_Alpha_trelis_TN_2D = (double *)(dev_Alpha_trelis_TNM_3D + (idx_m*dim1_Alpha*dim2_Alpha));;
	unsigned int *dev_Gamma_trellis_backtrace_TN_2D = (unsigned int *)(dev_Gamma_trellis_backtrace_TNM_3D + (idx_m*dim1_Gamma*dim2_Gamma));
	
	unsigned int* dev_O_obsSequence_1D = (unsigned int *)(dev_O_obsSequences_2D + idx_m*dim1_obsSeq);

	unsigned int* dev_likeliestStateIndexSequence_1D = (unsigned int *)(dev_likeliestStateIndexSequence_2D + (idx_m*T_noOfObservations));


	// --------------------------------------------------------------------------------------------------------
	// device memory initialization
	// --------------------------------------------------------------------------------------------------------

	TrellisInitialization2D(dev_Alpha_trelis_TN_2D, dev_Pi_startProbs_1D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	// ------------------------------------------------------------------------------------------------------
	// creating indices for viterbi1D
	// ------------------------------------------------------------------------------------------------------
	for (unsigned int idx_t = 1; idx_t < T_noOfObservations; idx_t++)
	{
		for (unsigned int idx_i = 0; idx_i < N_noOfStates; idx_i++)
		{
			for (unsigned int idx_j = 0; idx_j < N_noOfStates; idx_j++)
			{
				// ------------------------------------------------------------------------------------------------------
				// calling viterbi1D
				// ------------------------------------------------------------------------------------------------------
				viterbi1D(dev_Alpha_trelis_TN_2D, dev_Gamma_trellis_backtrace_TN_2D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, idx_i, idx_j, idx_t, dim1_Alpha, dim1_A, dim1_B);
			}
		}

		// ------------------------------------------------------------------------------------------------------
		// rescaling of Trellis' to avoid numerical problems
		// ------------------------------------------------------------------------------------------------------
		//TrellisScaling2D(dev_Alpha_trelis_TN_2D, T_noOfObservations, N_noOfStates, idx_t);
	}

	// ------------------------------------------------------------------------------------------------------
	// extract most likely path of states that generates observation
	// ------------------------------------------------------------------------------------------------------

	// backtrace(t,i) = j;

	/* backtrace */
	double partPathProb_optT = 0;
	unsigned int idx_j = 0;
	for (unsigned int idx_i = 0; idx_i < N_noOfStates; idx_i++) {
		//if (i == 0 || dev_Alpha_trelis_TN_2D[T_noOfObservations - 1][i] > partPathProb_optT) {
		if (idx_i == 0 || dev_Alpha_trelis_TN_2D[(T_noOfObservations - 1)*N_noOfStates + idx_i] > partPathProb_optT)
		{
			//partPathProb_optT = dev_Alpha_trelis_TN_2D[T_noOfObservations - 1][i];
			partPathProb_optT = dev_Alpha_trelis_TN_2D[(T_noOfObservations - 1)*N_noOfStates + idx_i];
			idx_j = idx_i;
		}
	}

	dev_likeliestStateIndexSequence_1D[T_noOfObservations - 1] = idx_j;
	for (unsigned int idx_i = 1; idx_i < T_noOfObservations; idx_i++)
	{
		//dev_likeliestStateIndexSequence_1D[T_noOfObservations - 1 - i] = dev_Gamma_trellis_backtrace_TN_2D[T_noOfObservations - i][dev_likeliestStateIndexSequence_1D[T_noOfObservations - i]];
		dev_likeliestStateIndexSequence_1D[T_noOfObservations - 1 - idx_i] = dev_Gamma_trellis_backtrace_TN_2D[(T_noOfObservations - idx_i)*N_noOfStates + dev_likeliestStateIndexSequence_1D[T_noOfObservations - idx_i]];
	}

}

// ------------------------------------------------------------------------------------------------------

__host__ __device__ void viterbi2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B)
{
	//#ifdef __CUDA_ARCH__
	//	printf("Device Thread %d\n", threadIdx.x);
	//#else
	//	printf("Host code!\n");
	//#endif

	// ------------------------------------------------------------------------------------------------------

	unsigned int idx_a_ji = 0;
	unsigned int idx_b_it = 0;
	unsigned int idx_p = 0;

	createViterbiIndices2DDevice(idx_a_ji, idx_b_it, idx_p, idx_i, idx_j, idx_t, dim1_P, dim2_P, dim1_A, dim1_B);

	// ------------------------------------------------------------------------------------------------------

	double a_ji = dev_A_stateTransProbs_2D[idx_a_ji];
	double b_it = dev_B_obsEmissionProbs_2D[idx_b_it];
	double p = a_ji * b_it;
	dev_probs_3D[idx_p] = p;
}

__global__ void viterbiKernel2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int T_noOfObservations, unsigned int idx_t, unsigned int V_noOfObsSymbols)
{
	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	unsigned int dim1_A = 0;
	unsigned int dim1_B = 0;
	unsigned int dim1_P = 0;
	unsigned int dim2_P = 0;

	createViterbiMatrixDimensions2DDevice(dim1_A, dim1_B, dim1_P, dim2_P, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// determine indices
	// ------------------------------------------------------------------------------------------------------

	unsigned int idx_i = blockIdx.x;
	unsigned int idx_j = threadIdx.x;
	//unsigned int t = idx_t;

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	viterbi2D(dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, idx_i, idx_j, idx_t, dim1_P, dim2_P, dim1_A, dim1_B);

}

// ------------------------------------------------------------------------------------------------------

__host__ __device__ void createViterbiIndices1D(unsigned int &idx_a_ji, unsigned int &idx_b_it, unsigned int &idx_alpha_ti, unsigned int &idx_alpha_tm1j,
	unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_A, unsigned int dim1_B, unsigned int dim1_Alpha)
{
//#ifdef	COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// COLUMN-MAJOR ORDER MATRIX: the first dimension in the array iterates the rows in the same column
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	// calculate transition and emmision index in 2D transition and emmision arrays of size dim1 * dim2:
	// a_ji
	idx_a_ji = idx_j + idx_i*dim1_A;
	// b_it
	idx_b_it = idx_i + idx_t*dim1_B;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	//idx_p = idx_j + idx_i*dim1_P + idx_t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(idx_t-1)idx_j * p
	idx_alpha_ti = idx_t + idx_i*dim1_Alpha;
	idx_alpha_tm1j = (idx_t - 1) + idx_j*dim1_Alpha;
	// ------------------------------------------------------------------------------------------------------
//#endif

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
	idx_a_ji = idx_j*dim1_A + idx_i;
	// b_it
	idx_b_it = idx_i*dim1_B + idx_t;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	//idx_p = idx_j*dim1_P + idx_i + idx_t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(idx_t-1)idx_j * p
	idx_alpha_ti = idx_t*dim1_Alpha + idx_i;
	idx_alpha_tm1j = (idx_t - 1)*dim1_Alpha + idx_j;

#endif
}

// ------------------------------------------------------------------------------------------------------

__host__ __device__ void createViterbiIndices2DDevice(unsigned int &idx_a_ji, unsigned int &idx_b_it, unsigned int &idx_p, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B)
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
	idx_a_ji = idx_j + idx_i*dim1_A;
	// b_it
	idx_b_it = idx_i + idx_t*dim1_B;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = idx_j + idx_i*dim1_P + idx_t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(idx_t-1)idx_j * p
	//idx_alpha_ti = idx_t + idx_i*dim1_Alpha;
	//idx_alpha_tm1j = (idx_t - 1) + idx_j*dim1_Alpha;
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
	idx_a_ji = idx_j*dim1_A + idx_i;
	// b_it
	idx_b_it = idx_i*dim1_B + idx_t;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = idx_j*dim1_P + idx_i + idx_t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(idx_t-1)idx_j * p
	//idx_alpha_ti = idx_t*dim1_Alpha + idx_i;
	//idx_alpha_tm1j = (idx_t - 1)*dim1_Alpha + idx_j;

#endif

}

__host__ void createViterbiIndices2DHost(unsigned int &idx_p, unsigned int &idx_alpha_ti, unsigned int &idx_alpha_tm1j, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_Alpha, unsigned int dim1_P, unsigned int dim2_P)
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
	//idx_a_ji = idx_j + idx_i*dim1_A;
	// b_it
	//idx_b_it = idx_i + idx_t*dim1_B;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = idx_j + idx_i*dim1_P + idx_t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(idx_t-1)idx_j * p
	idx_alpha_ti = idx_t + idx_i*dim1_Alpha;
	idx_alpha_tm1j = (idx_t - 1) + idx_j*dim1_Alpha;
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
	//idx_a_ji = idx_j*dim1_A + idx_i;
	// b_it
	//idx_b_it = idx_i*dim1_B + idx_t;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	idx_p = idx_j*dim1_P + idx_i + idx_t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(idx_t-1)idx_j * p
	idx_alpha_ti = idx_t*dim1_Alpha + idx_i;
	idx_alpha_tm1j = (idx_t - 1)*dim1_Alpha + idx_j;

#endif

}

// ------------------------------------------------------------------------------------------------------

__device__ void createViterbiMatrixDimensions1D(unsigned int &dim1_Alpha, unsigned int &dim2_Alpha, unsigned int &dim1_A, unsigned int &dim1_B, unsigned int T_noOfObservations, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols)
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
	dim2_Alpha = N_noOfStates;  // would be number of states (in the column) but not needed here

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
	dim2_Alpha = T_noOfObservations;  // would be size of observation sequence (in the column) but not needed here

	// ------------------------------------------------------------------------------------------------------
#endif

}

// ------------------------------------------------------------------------------------------------------

__device__ void createViterbiMatrixDimensions2DDevice(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols)
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

	dim1_P = blockDim.x;
	dim2_P = gridDim.x;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

#endif
}

__host__ void createViterbiMatrixDimensions2DHost(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_Alpha, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int N_noOfStates, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols)
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
// wrapper functions to switch transparently between GPU and CPU calcuation 
// without changing the main algorithms
// Call for the 3D Variant
// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ViterbiAlgorithm2D(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, unsigned int *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// choose environment of calculation
	// ------------------------------------------------------------------------------------------------------

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = ViterbiAlgorithm2DGPU(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_Gamma_trellis_backtrace_2D, host_probs_3D, host_likeliestStateIndexSequence_1D);
		break;
	case ComputationEnvironment::CPU:
		cudaStatus = ViterbiAlgorithm2DCPU(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_Gamma_trellis_backtrace_2D, host_probs_3D, host_likeliestStateIndexSequence_1D);
		break;
	}

	if (cudaStatus != cudaError_t::cudaSuccess)
		return cudaStatus;

	// ------------------------------------------------------------------------------------------------------
	// calculate Gamma/AlphaTrellis2D in a serial fashion
	// ------------------------------------------------------------------------------------------------------

	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	unsigned int dim1_A = 0;
	unsigned int dim1_B = 0;
	unsigned int dim1_Alpha = 0;
	unsigned int dim1_P = 0;
	unsigned int dim2_P = 0;

	createViterbiMatrixDimensions2DHost(dim1_A, dim1_B, dim1_Alpha, dim1_P, dim2_P, N_noOfStates, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------

	for (unsigned int idx_t = 1; idx_t < T_noOfObservations; idx_t++)
	{
		for (unsigned int idx_i = 0; idx_i < N_noOfStates; idx_i++)
		{
			for (unsigned int idx_j = 0; idx_j < N_noOfStates; idx_j++)
			{
				// ------------------------------------------------------------------------------------------------------
				// determine indices
				// ------------------------------------------------------------------------------------------------------

				unsigned int idx_p = 0;
				unsigned int idx_alpha_ti = 0;
				unsigned int idx_alpha_tm1j = 0;

				createViterbiIndices2DHost(idx_p, idx_alpha_ti, idx_alpha_tm1j, idx_i, idx_j, idx_t, dim1_Alpha, dim1_P, dim2_P);

				// ------------------------------------------------------------------------------------------------------
				// actual calculation
				// ------------------------------------------------------------------------------------------------------
				double p = host_probs_3D[idx_p];
				double alpha_tm1j = host_Alpha_trelis_2D[idx_alpha_tm1j];
				double alpha_ti = host_Alpha_trelis_2D[idx_alpha_ti];
				double partPathProb_tm1j = alpha_tm1j * p;
				if (partPathProb_tm1j >alpha_ti)
				{
					host_Alpha_trelis_2D[idx_alpha_ti] = partPathProb_tm1j;
					// backpointers(t,i) = j
					host_Gamma_trellis_backtrace_2D[idx_alpha_ti] = idx_j;
				}

			}
		}
	}

	// ------------------------------------------------------------------------------------------------------
	// extract most likely path of states that generates observation
	// ------------------------------------------------------------------------------------------------------

	//host_likeliestStateSequence = (unsigned int *)calloc(T_noOfObservations, sizeof(unsigned int));
	// backtrace(t,i) = j;

	/* backtrace */
	double partPathProb_optT = 0;
	unsigned int idx_j = 0;
	for (unsigned int idx_i = 0; idx_i < N_noOfStates; idx_i++) {
		//if (i == 0 || host_Alpha_trelis_2D[T_noOfObservations - 1][i] > partPathProb_optT) {
		if (idx_i == 0 || host_Alpha_trelis_2D[(T_noOfObservations - 1)*N_noOfStates + idx_i] > partPathProb_optT)
		{
			//partPathProb_optT = host_Alpha_trelis_2D[T_noOfObservations - 1][i];
			partPathProb_optT = host_Alpha_trelis_2D[(T_noOfObservations - 1)*N_noOfStates + idx_i];
			idx_j = idx_i;
		}
	}

	host_likeliestStateIndexSequence_1D[T_noOfObservations - 1] = idx_j;
	for (unsigned int idx_i = 1; idx_i < T_noOfObservations; idx_i++)
	{
		//host_likeliestStateSequence[T_noOfObservations - 1 - i] = host_Gamma_trellis_backtrace_2D[T_noOfObservations - i][host_likeliestStateSequence[T_noOfObservations - i]];
		host_likeliestStateIndexSequence_1D[T_noOfObservations - 1 - idx_i] = host_Gamma_trellis_backtrace_2D[(T_noOfObservations - idx_i)*N_noOfStates + host_likeliestStateIndexSequence_1D[T_noOfObservations - idx_i]];
	}

	
	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

__host__ cudaError_t ViterbiAlgorithm2DGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, unsigned int *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D)
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
		deviceFree(&dev_O_obsSequence_1D);
		return cudaStatus;
	}


	// ------------------------------------------------------------------------------------------------------
	// copy memory from host to device
	// ------------------------------------------------------------------------------------------------------


	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	for (unsigned int idx_t = 1; idx_t < T_noOfObservations; idx_t++){

		// call kernel for NxT matrix ops (N is the number of states, T is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.
		viterbiKernel2D << <N_noOfStates, N_noOfStates >> >(dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, T_noOfObservations, idx_t, V_noOfObsSymbols);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	// ------------------------------------------------------------------------------------------------------

	if (cudaStatus != cudaSuccess) {
		deviceFree(&dev_O_obsSequence_1D);
		deviceFree(&dev_probs_3D);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// copy memory from device to host
	// ------------------------------------------------------------------------------------------------------

	// Copy output vector from GPU buffer to host memory.
	if ((cudaStatus = memcpyVector(&host_probs_3D, dev_probs_3D, N_noOfStates * N_noOfStates * T_noOfObservations, cudaMemcpyDeviceToHost)) != cudaSuccess) {
		deviceFree(&dev_O_obsSequence_1D);
		deviceFree(&dev_probs_3D);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// ------------------------------------------------------------------------------------------------------
	deviceFree(&dev_probs_3D);

	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

__host__ cudaError_t ViterbiAlgorithm2DCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, unsigned int *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	unsigned int dim1_A = 0;
	unsigned int dim1_B = 0;
	unsigned int dim1_Alpha = 0;
	unsigned int dim1_P = 0;
	unsigned int dim2_P = 0;

	createViterbiMatrixDimensions2DHost(dim1_A, dim1_B, dim1_Alpha, dim1_P, dim2_P, N_noOfStates, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// determine indices
	// ------------------------------------------------------------------------------------------------------

	for (unsigned int idx_t = 1; idx_t < T_noOfObservations; idx_t++){

		// call kernel for NxT matrix ops (N is the number of states, T is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.

		//forwardKernel << <N_noOfStates, N_noOfStates >> >(dev_Alpha_trelis_2D, dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, T_noOfObservations, idx_t, V_noOfObsSymbols);
		for (unsigned int i = 0; i < N_noOfStates; i++)
		{
			for (unsigned int j = 0; j < N_noOfStates; j++)
			{
				// ------------------------------------------------------------------------------------------------------
				// actual calculation
				// ------------------------------------------------------------------------------------------------------

				// make kernel callable as normal function on host
				viterbi2D(host_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, i, j, idx_t, dim1_P, dim2_P, dim1_A, dim1_B);

			}
		}
	}

	return cudaStatus;
}

// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ViterbiAlgorithm1DCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, unsigned int *host_Gamma_trellis_backtrace_2D, unsigned int *host_likeliestStateIndexSequence_1D)
{
	cudaError_t cudaStatus = cudaError_t::cudaSuccess;

	// ------------------------------------------------------------------------------------------------------
	// calculate Gamma/AlphaTrellis2D in a serial fashion
	// ------------------------------------------------------------------------------------------------------

	// ------------------------------------------------------------------------------------------------------
	// determine matrix dimensions
	// ------------------------------------------------------------------------------------------------------

	unsigned int dim1_A = 0;
	unsigned int dim1_B = 0;
	unsigned int dim1_Alpha = 0;
	unsigned int dim1_P = 0;
	unsigned int dim2_P = 0;

	createViterbiMatrixDimensions2DHost(dim1_A, dim1_B, dim1_Alpha, dim1_P, dim2_P, N_noOfStates, T_noOfObservations, V_noOfObsSymbols);

	// ------------------------------------------------------------------------------------------------------
	// HACK - directly use host pointers instead of device pointers via memory allocation routines

	double *dev_Alpha_trelis_2D = host_Alpha_trelis_2D;
	unsigned int *dev_Gamma_trellis_backtrace_2D = host_Gamma_trellis_backtrace_2D;

	// ------------------------------------------------------------------------------------------------------

	for (unsigned int idx_t = 1; idx_t < T_noOfObservations; idx_t++)
	{
		// ------------------------------------------------------------------------------------------------------
		// creating indices for viterbi1D
		// ------------------------------------------------------------------------------------------------------
		for (unsigned int idx_i = 0; idx_i < N_noOfStates; idx_i++)
		{
			for (unsigned int idx_j = 0; idx_j < N_noOfStates; idx_j++)
			{
				// ------------------------------------------------------------------------------------------------------
				// calling viterbi1D
				// ------------------------------------------------------------------------------------------------------
				viterbi1D(dev_Alpha_trelis_2D, dev_Gamma_trellis_backtrace_2D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, idx_i, idx_j, idx_t, dim1_Alpha, dim1_A, dim1_B);
			}
		}

		// ------------------------------------------------------------------------------------------------------
		// rescaling of Trellis' to avoid numerical problems
		// ------------------------------------------------------------------------------------------------------
		TrellisScaling2D(dev_Alpha_trelis_2D, T_noOfObservations, N_noOfStates, idx_t);
	}

	// ------------------------------------------------------------------------------------------------------
	// extract most likely path of states that generates observation
	// ------------------------------------------------------------------------------------------------------

	// backtrace(t,i) = j;

	/* backtrace */
	double partPathProb_optT = 0;
	unsigned int idx_j = 0;
	for (unsigned int idx_i = 0; idx_i < N_noOfStates; idx_i++) {
		//if (i == 0 || host_Alpha_trelis_2D[T_noOfObservations - 1][i] > partPathProb_optT) {
		if (idx_i == 0 || host_Alpha_trelis_2D[(T_noOfObservations - 1)*N_noOfStates + idx_i] > partPathProb_optT)
		{
			//partPathProb_optT = host_Alpha_trelis_2D[T_noOfObservations - 1][i];
			partPathProb_optT = host_Alpha_trelis_2D[(T_noOfObservations - 1)*N_noOfStates + idx_i];
			idx_j = idx_i;
		}
	}

	host_likeliestStateIndexSequence_1D[T_noOfObservations - 1] = idx_j;
	for (unsigned int idx_i = 1; idx_i < T_noOfObservations; idx_i++)
	{
		//host_likeliestStateSequence[T_noOfObservations - 1 - i] = host_Gamma_trellis_backtrace_2D[T_noOfObservations - i][host_likeliestStateSequence[T_noOfObservations - i]];
		host_likeliestStateIndexSequence_1D[T_noOfObservations - 1 - idx_i] = host_Gamma_trellis_backtrace_2D[(T_noOfObservations - idx_i)*N_noOfStates + host_likeliestStateIndexSequence_1D[T_noOfObservations - idx_i]];
	}

	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ViterbiAlgorithmSet1D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D,
	unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, unsigned int *host_likeliestStateIndexSequence_2D)
{
	cudaError_t cudaStatus = cudaSuccess;
	// ------------------------------------------------------------------------------------------------------
	// choose environment of calculation
	// ------------------------------------------------------------------------------------------------------

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = ViterbiAlgorithmSet1DGPU(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D,
			N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likeliestStateIndexSequence_2D);
		break;
	case ComputationEnvironment::CPU:
		cudaStatus = ViterbiAlgorithmSet1DCPU(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D,
			N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likeliestStateIndexSequence_2D);
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t ViterbiAlgorithmSet1DGPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, 
	unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, unsigned int *host_likeliestStateIndexSequence_2D)
{
	double *dev_Pi_startProbs_1D = nullptr;
	double *dev_A_stateTransProbs_2D = nullptr;
	double *dev_B_obsEmissionProbs_2D = nullptr;
	double *dev_Alpha_trelis_TNM_3D = nullptr;
	unsigned int *dev_Gamma_trellis_backtrace_TNM_3D = nullptr;
	unsigned int *dev_likeliestStateIndexSequence_2D = nullptr;
	unsigned int *dev_O_obsSequences_2D = nullptr;

	cudaError_t cudaStatus = cudaSuccess;

	// --------------------------------------------------------------------------------------------------------
	// device memory allocation
	// --------------------------------------------------------------------------------------------------------

	if ((cudaStatus = allocateDeviceVector(&dev_Pi_startProbs_1D, N_noOfStates)) != cudaSuccess) {
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_A_stateTransProbs_2D, N_noOfStates*N_noOfStates)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_Alpha_trelis_TNM_3D, T_noOfObservations*N_noOfStates*M_noOfObsSequences, true)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}
	if ((cudaStatus = allocateDeviceVector(&dev_Gamma_trellis_backtrace_TNM_3D, T_noOfObservations*N_noOfStates*M_noOfObsSequences, true)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_likeliestStateIndexSequence_2D, M_noOfObsSequences*T_noOfObservations, true)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_O_obsSequences_2D, M_noOfObsSequences*T_noOfObservations, true)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
		deviceFree(&dev_likeliestStateIndexSequence_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------------------
	// memory copy from host to device
	// --------------------------------------------------------------------------------------------------------
	// Copy input vectors from host memory to GPU buffers.
	if ((cudaStatus = memcpyVector(&dev_Pi_startProbs_1D, (double *)host_Pi_startProbs_1D, N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
		deviceFree(&dev_likeliestStateIndexSequence_2D);
		deviceFree(&dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(&dev_A_stateTransProbs_2D, (double *)host_A_stateTransProbs_2D, N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
		deviceFree(&dev_likeliestStateIndexSequence_2D);
		deviceFree(&dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(&dev_B_obsEmissionProbs_2D, (double *)host_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
		deviceFree(&dev_likeliestStateIndexSequence_2D);
		deviceFree(&dev_O_obsSequences_2D);
		return cudaStatus;
	}


	if ((cudaStatus = memcpyVector(&dev_O_obsSequences_2D, (unsigned int *)host_O_obsSequences_2D, M_noOfObsSequences*T_noOfObservations, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
		deviceFree(&dev_likeliestStateIndexSequence_2D);
		deviceFree(&dev_O_obsSequences_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX
		
	unsigned int dimBlock = glob_blocksize;
	unsigned int dimGrid = ceil(M_noOfObsSequences / (float)dimBlock);
	//// HACK
	//unsigned int dimBlock = M_noOfObsSequences;
	//unsigned int dimGrid = 1;

	viterbiKernel1D << <dimGrid, dimBlock >> >(dev_Pi_startProbs_1D, dev_Alpha_trelis_TNM_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D,
		dev_O_obsSequences_2D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols, M_noOfObsSequences, dev_Gamma_trellis_backtrace_TNM_3D, dev_likeliestStateIndexSequence_2D);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	
	// --------------------------------------------------------------------------------------------------------
	// memory copy from device to host
	// --------------------------------------------------------------------------------------------------------

	if ((cudaStatus = memcpyVector(&host_likeliestStateIndexSequence_2D, dev_likeliestStateIndexSequence_2D, M_noOfObsSequences*T_noOfObservations, cudaMemcpyDeviceToHost)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		deviceFree(&dev_Alpha_trelis_TNM_3D);
		deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
		deviceFree(&dev_likeliestStateIndexSequence_2D);
		deviceFree(&dev_O_obsSequences_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------
	// host memory cleanup
	// --------------------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// --------------------------------------------------------------------------------------------------------
	deviceFree(&dev_Pi_startProbs_1D);
	deviceFree(&dev_A_stateTransProbs_2D);
	deviceFree(&dev_B_obsEmissionProbs_2D);
	deviceFree(&dev_Alpha_trelis_TNM_3D);
	deviceFree(&dev_Gamma_trellis_backtrace_TNM_3D);
	deviceFree(&dev_likeliestStateIndexSequence_2D);
	deviceFree(&dev_O_obsSequences_2D);
	// --------------------------------------------------------------------------------------------------------

#endif

}

__host__ cudaError_t ViterbiAlgorithmSet1DCPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, unsigned int *host_likeliestStateIndexSequence_2D)
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
		deviceFree(&dev_Pi_startProbs_1D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------------------
	// memory copy from host do device
	// --------------------------------------------------------------------------------------------------------
	// Copy input vectors from host memory to GPU buffers.
	if ((cudaStatus = memcpyVector(&dev_Pi_startProbs_1D, (double *)host_Pi_startProbs_1D, N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(&dev_A_stateTransProbs_2D, (double *)host_A_stateTransProbs_2D, N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(&dev_B_obsEmissionProbs_2D, (double *)host_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// for each obs. sequence do
	for (unsigned int idx_m = 0; idx_m<M_noOfObsSequences; idx_m++) {

		//cout << "starting viterbi alg for obs sequence...\n";

		// --------------------------------------------------------------------------------------------------------
		// host memory allocation
		// --------------------------------------------------------------------------------------------------------
		double* host_Alpha_trelis_2D = (double *)calloc(T_noOfObservations * N_noOfStates, sizeof(double));
		unsigned int* host_Gamma_trellis_backtrace_2D = (unsigned int *)calloc(T_noOfObservations * N_noOfStates, sizeof(double));

		unsigned int* host_likeliestStateIndexSequence_1D = (unsigned int *)(host_likeliestStateIndexSequence_2D + (idx_m*T_noOfObservations));

		// --------------------------------------------------------------------------------------------------------
		
		// extract the right pointer position out of host_O_obsSequences_2D
		unsigned int dim1_M = T_noOfObservations;
		unsigned int* host_O_obsSequence_1D = nullptr;
		host_O_obsSequence_1D = (unsigned int *)(host_O_obsSequences_2D + (idx_m*dim1_M)); // seems to be ok

		// --------------------------------------------------------------------------------------------------------
		// host memory initialization
		// --------------------------------------------------------------------------------------------------------

		TrellisInitialization2D(host_Alpha_trelis_2D, host_Pi_startProbs_1D, host_B_obsEmissionProbs_2D, host_O_obsSequence_1D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

		// --------------------------------------------------------------------------------------------------------
		// actual calculation
		// --------------------------------------------------------------------------------------------------------

		double host_likelihood = 0;
		cudaError_t cudaStatus = ViterbiAlgorithm1DCPU(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_Gamma_trellis_backtrace_2D, host_likeliestStateIndexSequence_1D);

		if (cudaStatus != cudaSuccess) {
			deviceFree(&dev_Pi_startProbs_1D);
			deviceFree(&dev_A_stateTransProbs_2D);
			deviceFree(&dev_B_obsEmissionProbs_2D);
			return cudaStatus;
		}

		// --------------------------------------------------------------------------------------------------------
		// extract likeliestStateSequence
		// --------------------------------------------------------------------------------------------------------
		// already done by pointers
		//host_likeliestStateSequence_2D[] = host_likeliestStateSequence_1D;

		// --------------------------------------------------------------------------------------------------------
		// host memory cleanup
		// --------------------------------------------------------------------------------------------------------
		free(host_Alpha_trelis_2D);
		free(host_Gamma_trellis_backtrace_2D);
		// --------------------------------------------------------------------------------------------------------
	}

	// --------------------------------------------------------------------------------------------------------
	// host memory cleanup
	// --------------------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// --------------------------------------------------------------------------------------------------------
	deviceFree(&dev_Pi_startProbs_1D);
	deviceFree(&dev_A_stateTransProbs_2D);
	deviceFree(&dev_B_obsEmissionProbs_2D);

	// --------------------------------------------------------------------------------------------------------

#endif

}

// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ViterbiAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, unsigned int *host_likeliestStateIndexSequence_2D, bool printToConsole)
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
		deviceFree(&dev_Pi_startProbs_1D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------------------
	// memory copy from host do device
	// --------------------------------------------------------------------------------------------------------
	// Copy input vectors from host memory to GPU buffers.
	if ((cudaStatus = memcpyVector(&dev_Pi_startProbs_1D, (double *)host_Pi_startProbs_1D, N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(&dev_A_stateTransProbs_2D, (double *)host_A_stateTransProbs_2D, N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(&dev_B_obsEmissionProbs_2D, (double *)host_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(&dev_Pi_startProbs_1D);
		deviceFree(&dev_A_stateTransProbs_2D);
		deviceFree(&dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// TODO: the M different observation sequences could be computed in parallel

	// allocated outside
	//unsigned int* host_likeliestStateIndexSequence_2D = (unsigned int *)calloc(M_noOfObsSequences*T_noOfObservations, sizeof(unsigned int));

	// for each obs. sequence do
	for (unsigned int i = 0; i<M_noOfObsSequences; i++) {

		if (printToConsole)
			cout << "starting viterbi alg for obs sequence...\n";

		// --------------------------------------------------------------------------------------------------------
		// host memory allocation
		// --------------------------------------------------------------------------------------------------------
		double* host_Alpha_trelis_2D = (double *)calloc(T_noOfObservations * N_noOfStates, sizeof(double));
		unsigned int* host_Gamma_trellis_backtrace_2D = (unsigned int *)calloc(T_noOfObservations * N_noOfStates, sizeof(double));

		double* host_probs_3D = (double *)calloc(N_noOfStates * N_noOfStates * T_noOfObservations, sizeof(double));
		//unsigned int* host_likeliestStateIndexSequence_1D = (unsigned int *)calloc(T_noOfObservations, sizeof(unsigned int));
		unsigned int* host_likeliestStateIndexSequence_1D = (unsigned int *)(host_likeliestStateIndexSequence_2D + (i*T_noOfObservations));

		// extract the right pointer position out of host_O_obsSequences_2D
		unsigned int dim1_M = T_noOfObservations;
		unsigned int* host_O_obsSequence_1D = nullptr;
		host_O_obsSequence_1D = (unsigned int *)(host_O_obsSequences_2D + (i*dim1_M)); // seems to be ok

		// --------------------------------------------------------------------------------------------------------
		// host memory initialization
		// --------------------------------------------------------------------------------------------------------

		TrellisInitialization2D(host_Alpha_trelis_2D, host_Pi_startProbs_1D, host_B_obsEmissionProbs_2D, host_O_obsSequence_1D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

		// --------------------------------------------------------------------------------------------------------
		// actual calculation
		// --------------------------------------------------------------------------------------------------------

		double host_likelihood = 0;
		cudaError_t cudaStatus = ViterbiAlgorithm2D(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_Gamma_trellis_backtrace_2D, host_probs_3D, host_likeliestStateIndexSequence_1D);

		if (cudaStatus != cudaSuccess) {
			deviceFree(&dev_Pi_startProbs_1D);
			deviceFree(&dev_A_stateTransProbs_2D);
			deviceFree(&dev_B_obsEmissionProbs_2D);
			return cudaStatus;
		}

		// --------------------------------------------------------------------------------------------------------
		// extract likeliestStateSequence
		// --------------------------------------------------------------------------------------------------------
		// already done by pointers
		//host_likeliestStateSequence_2D[] = host_likeliestStateSequence_1D;

		// --------------------------------------------------------------------------------------------------------
		// host memory cleanup
		// --------------------------------------------------------------------------------------------------------
		free(host_Alpha_trelis_2D);
		free(host_Gamma_trellis_backtrace_2D);
		free(host_probs_3D);
		//free(host_likeliestStateIndexSequence_1D);
		// --------------------------------------------------------------------------------------------------------
	}

	// --------------------------------------------------------------------------------------------------------
	// host memory cleanup
	// --------------------------------------------------------------------------------------------------------
	// allocated outside
	//free(host_likeliestStateIndexSequence_2D);

	// --------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// --------------------------------------------------------------------------------------------------------
	deviceFree(&dev_Pi_startProbs_1D);
	deviceFree(&dev_A_stateTransProbs_2D);
	deviceFree(&dev_B_obsEmissionProbs_2D);

	// --------------------------------------------------------------------------------------------------------

#endif

}

