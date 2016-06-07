#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MemoryManagement.cuh"

#include "Matricies.h"
#include "Observation.h"

//#define COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
#define ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

// ------------------------------------------------------------------------------------------------------
// global states
// ------------------------------------------------------------------------------------------------------
extern ComputationEnvironment glob_Env;

// ------------------------------------------------------------------------------------------------------
// forward declarations
// ------------------------------------------------------------------------------------------------------

__global__ void fwKernel(double *p, const double *transition, const double *emission, int obs);
__global__ void forwardKernel(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int T_noOfObservations, int idx_obs, int V_noOfObsSymbols);

//__host__ cudaError_t ForwardAlgorithm(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D, double &likelihood);
//__host__ cudaError_t ForwardAlgorithmGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D, double &likelihood);
//__host__ cudaError_t ForwardAlgorithmCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D, double &likelihood);

__host__ cudaError_t ForwardAlgorithm(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ForwardAlgorithmGPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ForwardAlgorithmCPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);

// ------------------------------------------------------------------------------------------------------


int main(int argc, char* argv[])
{

	cout << "start...\n";

	cudaError_t cudaStatus;
	double *host_Pi_startProbs_1D = nullptr;
	double *host_A_stateTransProbs_2D = nullptr;
	double *host_B_obsEmissionProbs_2D = nullptr;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	Matricies* matricies = new Matricies();
	Observation* observations = new Observation();
	int N_noOfStates = matricies->N;
	int V_noOfObsSymbols = matricies->V;

	matricies->loadMatricies(argv[1]);
	observations->loadObservations(argv[1]);

	// --------------------------------------------------------------------------------------------------------

	host_Pi_startProbs_1D = matricies->piAsArray();
	host_A_stateTransProbs_2D = matricies->transitionAsArray();
	host_B_obsEmissionProbs_2D = matricies->emissionAsArray();

	// --------------------------------------------------------------------------------------------------------


	vector<vector<unsigned int>*>* sequences = &observations->sequences;
	int numberOfObservations = sequences->size();

	// for each obs. sequence do
	for (unsigned int i = 0; i<numberOfObservations; i++) {

		cout << "starting fw alg for obs sequence...\n";

		vector<unsigned int>* O_obsSequence = sequences->at(i);
		int T_noOfObservations = O_obsSequence->size();

		// --------------------------------------------------------------------------------------------------------
		double* host_Alpha_trelis_2D = (double *)calloc(T_noOfObservations * N_noOfStates, sizeof(double));
		double* host_probs_3D = (double *)calloc(N_noOfStates * N_noOfStates * T_noOfObservations, sizeof(double));
		unsigned int *host_O_obsSequence_1D = O_obsSequence->data();
		
		// --------------------------------------------------------------------------------------------------------

		double host_likelihood = 0;
		cudaStatus = ForwardAlgorithm(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_probs_3D, host_likelihood);
		
		// --------------------------------------------------------------------------------------------------------

		if (cudaStatus != cudaSuccess) {
			return cudaStatus;
		}

		// --------------------------------------------------------------------------------------------------------
		free(host_Alpha_trelis_2D);
		free(host_probs_3D);

		// --------------------------------------------------------------------------------------------------------
	}


	cout << "end\n";

	return 0;
}

// ------------------------------------------------------------------------------------------------------

__global__ void forwardKernel(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int T_noOfObservations, int idx_obs, int V_noOfObsSymbols)
{
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 2D-Grid, but called as 1D-Grid
	// ------------------------------------------------------------------------------------------------------
	//int ix = blockDim.x*blockIdx.x + threadIdx.x; // i
	//int iy = blockDim.y*blockIdx.y + threadIdx.y; // j

	//int idx_trans = iy * blockDim.x + ix; // blockDim.x == blockDim.y, cuda_2.pdf s.31
	//int idx_emit = ix * blockDim.x + obs;
	//int idx_prob = blockDim.x * blockDim.y * obs + blockDim.x * ix + iy;

#ifdef	COL_MAJ_ORD_MAT_ROW_FIRST_INDEX
	// ------------------------------------------------------------------------------------------------------
	// Indexing for 1D-Grid, called as 1D-Grid
	// COLUMN-MAJOR ORDER MATRIX: the first dimension in the array iterates the rows in the same column
	// ROW FIRST INDEXING: matrix indices starts with row i then column j A(i,j) 
	// ------------------------------------------------------------------------------------------------------
	// reference implementation: int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// vector layout: (i,j,t)

	int i = blockIdx.x;
	int j = threadIdx.x;
	int t = idx_obs;
	
	int dim1_A = blockDim.x;
	//int dim2_A = gridDim.x; // would be number of states (in the row) but not needed here
	
	int dim1_B = blockDim.x;
	//int dim2_B = V_noOfObsSymbols; // would be number of observation symbols but not needed here

	int dim1_Alpha = T_noOfObservations; // size of observation sequence
	//int dim2_Alpha = blockDim.x;  // would be number of states (in the column) but not needed here

	int dim1_P = blockDim.x;
	int dim2_P = gridDim.x;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

	// calculate transition and emmision index in 2D transition and emmision arrays of size dim1 * dim2:
	// a_ji
	int idx_a_ji = j + i*dim1_A;
	// b_it
	int idx_b_it = i + t*dim1_B;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	int idx_p = j + i*dim1_P + t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(t-1)j * p
	int idx_alpha_ti = t + i*dim1_Alpha;
	int idx_alpha_tm1j = (t-1) + j*dim1_Alpha;
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

	int i = blockIdx.x;
	int j = threadIdx.x;
	int t = idx_obs;

	int dim1_A = gridDim.x; // number of states (in the row)
	//int dim2_A = blockDim.x; // would be number of states (in the column) but not needed here

	int dim1_B = V_noOfObsSymbols; // number of observation symbols
	//int dim2_B =  blockDim.x; // would be number of states (in the column) but not needed here

	int dim1_Alpha = blockDim.x; // number of states (in the row)
	//int dim2_Alpha = T_noOfObservations;  // would be size of observation sequence (in the column) but not needed here

	int dim1_P = blockDim.x;
	int dim2_P = gridDim.x;
	//int dim3_P = T_noOfObservations; // would be number of observations but not needed here

	// calculate transition and emmision index in 2D transition and emmision arrays of size dim1 * dim2:
	// a_ji
	int idx_a_ji = j*dim1_A + i;
	// b_it
	int idx_b_it = i*dim1_B + t;
	// calculate probability index of 3D probability array of size dim1 * dim2 * dim3:
	// p = a_ji * b_it ... only temporary value, maybe p_jit ???
	int idx_p = j*dim1_P + i + t*dim1_P*dim2_P;
	// calculate alpha index of 2D trellis array of size dim1 * dim3:
	// alpha_ti = alpha_ti + alpha_(t-1)j * p
	int idx_alpha_ti = t*dim1_Alpha + i;
	int idx_alpha_tm1j = (t - 1)*dim1_Alpha + j;

	// ------------------------------------------------------------------------------------------------------
#endif

	double a_ji = dev_A_stateTransProbs_2D[idx_a_ji];
	double b_it = dev_B_obsEmissionProbs_2D[idx_b_it];
	double p = a_ji * b_it;
	dev_probs_3D[idx_p] = p;
	dev_Alpha_trelis_2D[idx_alpha_ti] = dev_Alpha_trelis_2D[idx_alpha_ti] + dev_Alpha_trelis_2D[idx_alpha_tm1j] * p;
}

__global__ void fwKernel(double *p, const double *transition, const double *emission, int obs){

	int ix = blockDim.x*blockIdx.x + threadIdx.x; // i
	int iy = blockDim.y*blockIdx.y + threadIdx.y; // j

	int idx_trans = iy * blockDim.x + ix; // blockDim.x == blockDim.y, cuda_2.pdf s.31
	int idx_emit = ix * blockDim.x + obs;
	int idx_prob = blockDim.x * blockDim.y * obs + blockDim.x * ix + iy;

	double trans = transition[idx_trans];
	double emis = emission[idx_emit];
	p[idx_prob] = trans * emis;


}

// ------------------------------------------------------------------------------------------------------
// wrapper functions to switch transparently between GPU and CPU calcuation 
// without changing the main algorithms
// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ForwardAlgorithmGPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood)
{
	// ------------------------------------------------------------------------------------------------------
	// copy memory from host to device
	// ------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;
	double *dev_Pi_startProbs_1D = nullptr;
	double *dev_A_stateTransProbs_2D = nullptr;
	double *dev_B_obsEmissionProbs_2D = nullptr;
	double *dev_probs_3D = nullptr;
	int *dev_O_obsSequence_1D = nullptr;

	// --------------------------------------------------------------------------------------------------------
	// memory allocation
	// --------------------------------------------------------------------------------------------------------

	// move this to ForwardAlgorithm
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

	// array to store the observation sequence
	if ((cudaStatus = allocateDeviceVector(&dev_O_obsSequence_1D, T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	// array to store all probabilities.
	if ((cudaStatus = allocateDeviceVector(&dev_probs_3D, N_noOfStates * N_noOfStates * T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequence_1D);
		return cudaStatus;
	}

	// array to store the trellis
	double *dev_Alpha_trelis_2D = nullptr;
	if ((cudaStatus = allocateDeviceVector(&dev_Alpha_trelis_2D, T_noOfObservations * N_noOfStates, true)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequence_1D);
		deviceFree(dev_probs_3D);
		deviceFree(dev_Alpha_trelis_2D);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// memory copy
	// ------------------------------------------------------------------------------------------------------
	// move this to ForwardAlgorithm
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

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	
	for (unsigned int idx_obs = 1; idx_obs < T_noOfObservations; idx_obs++){
		
		// call kernel for NxT matrix ops (N is the number of states, T is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.
		//forwardKernel << <dim3(N_noOfStates, N_noOfStates), dim3(N_noOfStates, N_noOfStates) >> >(dev_Alpha_trelis_2D, dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, obs);
		forwardKernel << <N_noOfStates, N_noOfStates >> >(dev_Alpha_trelis_2D, dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, T_noOfObservations, idx_obs, V_noOfObsSymbols);

	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	// ------------------------------------------------------------------------------------------------------

	if (cudaStatus != cudaSuccess) {
		// move this partly to ForwardAlgorithm
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequence_1D);
		deviceFree(dev_probs_3D);
		deviceFree(dev_Alpha_trelis_2D);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// copy memory from device to host
	// ------------------------------------------------------------------------------------------------------

	// Copy output vector from GPU buffer to host memory.
	if ((cudaStatus = memcpyVector(host_probs_3D, dev_probs_3D, N_noOfStates * N_noOfStates * T_noOfObservations, cudaMemcpyDeviceToHost)) != cudaSuccess) {
		// move this partly to ForwardAlgorithm
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequence_1D);
		deviceFree(dev_probs_3D);
		deviceFree(dev_Alpha_trelis_2D);
		return cudaStatus;
	}

	// ------------------------------------------------------------------------------------------------------
	// memory cleanup
	// ------------------------------------------------------------------------------------------------------
	deviceFree(dev_probs_3D);
	deviceFree(dev_Alpha_trelis_2D);

	// ------------------------------------------------------------------------------------------------------
	// move this to  ForwardAlgorithm
	deviceFree(dev_Pi_startProbs_1D);
	deviceFree(dev_A_stateTransProbs_2D);
	deviceFree(dev_B_obsEmissionProbs_2D);

	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithmCPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// actual calculation
	// ------------------------------------------------------------------------------------------------------

	// call kernel for NxT matrix ops (N is the number of states, T is the number of observations)
	// Launch a kernel on the GPU with one thread for each element.
	//	fwKernel << <N, N >> >(dev_probability, dev_transition, dev_emission, i);
	for (int i = 0; i < N_noOfStates; i++)
	{
		for (int j = 0; j < N_noOfStates; j++)
		{

		}
	}

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithm(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

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
		int idx_alpha_0i = i;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif

	// ------------------------------------------------------------------------------------------------------
	// choose environment of calculation
	// ------------------------------------------------------------------------------------------------------

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = ForwardAlgorithmGPU(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_probs_3D, host_likelihood);
		break;
	case ComputationEnvironment::CPU:
		cudaStatus = ForwardAlgorithmCPU(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, host_Alpha_trelis_2D, host_probs_3D, host_likelihood);
		break;
	}

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
	// ------------------------------------------------------------------------------------------------------

	return cudaStatus;
}

