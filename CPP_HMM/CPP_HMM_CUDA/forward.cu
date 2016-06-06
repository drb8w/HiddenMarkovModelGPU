#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matricies.h"
#include "Observation.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

// ------------------------------------------------------------------------------------------------------
// global states
// ------------------------------------------------------------------------------------------------------

enum ComputationEnvironment{CPU, GPU};

ComputationEnvironment glob_Env = ComputationEnvironment::GPU;

// ------------------------------------------------------------------------------------------------------
// typedefs
// ------------------------------------------------------------------------------------------------------
typedef int *IntPtr;
typedef IntPtr *IntHdl;

typedef float *FloatPtr;
typedef FloatPtr *FloatHdl; 

typedef double *DoublePtr;
typedef DoublePtr *DoubleHdl;

// ------------------------------------------------------------------------------------------------------
// forward declarations
// ------------------------------------------------------------------------------------------------------

__global__ void fwKernel(double *p, const double *transition, const double *emission, int obs);
__global__ void forwardKernel(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, int obs);

__host__ cudaError_t ForwardAlgorithm(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D);
__host__ cudaError_t ForwardAlgorithmGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D);
__host__ cudaError_t ForwardAlgorithmCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D);

__host__ cudaError_t allocateDeviceVector(IntHdl pVector, int numberOfElements);
__host__ cudaError_t allocateDeviceVector(FloatHdl pVector, int numberOfElements);
__host__ cudaError_t allocateDeviceVector(DoubleHdl pVector, int numberOfElements);

__host__ cudaError_t memcpyVector(IntPtr dst, const IntPtr src, int numberOfElements, enum cudaMemcpyKind kind);
__host__ cudaError_t memcpyVector(FloatPtr dst, const FloatPtr src, int numberOfElements, enum cudaMemcpyKind kind);
__host__ cudaError_t memcpyVector(DoublePtr dst, const DoublePtr src, int numberOfElements, enum cudaMemcpyKind kind);

__host__ cudaError_t deviceFree(void *devPtr);

// ------------------------------------------------------------------------------------------------------


int main(int argc, char* argv[])
{

	cout << "start...\n";

	cudaError_t cudaStatus;
	double *dev_A_stateTransProbs_2D = nullptr;
	double *dev_B_obsEmissionProbs_2D = nullptr;
	double *dev_probs_3D = nullptr;
	double *dev_Pi_startProbs_1D = nullptr;
	int *dev_O_obsSequence_1D = nullptr;

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
	// Copy input vectors from host memory to GPU buffers.
	if ((cudaStatus = memcpyVector(dev_Pi_startProbs_1D, matricies->piAsArray(), N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_A_stateTransProbs_2D, matricies->transitionAsArray(), N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D); 
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_B_obsEmissionProbs_2D, matricies->emissionAsArray(), N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D); 
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------

	vector<vector<unsigned int>*>* sequences = &observations->sequences;
	int numberOfObservations = sequences->size();

	// for each obs. sequence do
	for (unsigned int i = 0; i<numberOfObservations; i++) {

		cout << "starting fw alg for obs sequence...\n";

		vector<unsigned int>* O_obsSequence = sequences->at(i);
		int T_noOfObservations = O_obsSequence->size();

		double* host_probs_3D = new double[N_noOfStates * N_noOfStates * T_noOfObservations];

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
		if ((cudaStatus = allocateDeviceVector(&dev_Alpha_trelis_2D, T_noOfObservations * N_noOfStates)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D); 
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequence_1D);
			deviceFree(dev_probs_3D);
			deviceFree(dev_Alpha_trelis_2D);
			return cudaStatus;
		}

		// --------------------------------------------------------------------------------------------------------

		cudaStatus = ForwardAlgorithm(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, dev_Alpha_trelis_2D, dev_probs_3D);
		
		// --------------------------------------------------------------------------------------------------------

		if (cudaStatus != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D); 
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequence_1D);
			deviceFree(dev_probs_3D);
			deviceFree(dev_Alpha_trelis_2D);
			return cudaStatus;
		}

		// Copy output vector from GPU buffer to host memory.
		if ((cudaStatus = memcpyVector(host_probs_3D, dev_probs_3D, N_noOfStates * N_noOfStates * T_noOfObservations, cudaMemcpyDeviceToHost)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D); 
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequence_1D);
			deviceFree(dev_probs_3D);
			deviceFree(dev_Alpha_trelis_2D);
			return cudaStatus;
		}

		delete[] host_probs_3D;
		deviceFree(dev_probs_3D);
		deviceFree(dev_Alpha_trelis_2D);

	}

	deviceFree(dev_Pi_startProbs_1D);
	deviceFree(dev_A_stateTransProbs_2D);
	deviceFree(dev_B_obsEmissionProbs_2D);

	cout << "end\n";

	return 0;
}

// ------------------------------------------------------------------------------------------------------

__global__ void forwardKernel(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, int obs)
{

	int ix = blockDim.x*blockIdx.x + threadIdx.x; // i
	int iy = blockDim.y*blockIdx.y + threadIdx.y; // j

	int idx_trans = iy * blockDim.x + ix; // blockDim.x == blockDim.y, cuda_2.pdf s.31
	int idx_emit = ix * blockDim.x + obs;
	int idx_prob = blockDim.x * blockDim.y * obs + blockDim.x * ix + iy;

	double trans = dev_A_stateTransProbs_2D[idx_trans];
	double emis = dev_B_obsEmissionProbs_2D[idx_emit];
	dev_probs_3D[idx_prob] = trans * emis;
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

__host__ cudaError_t ForwardAlgorithmGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D)
{
	// ------------------------------------------------------------------------------------------------------
	// Initialization of the Alpha_trelis
	// ------------------------------------------------------------------------------------------------------
	// a_0i = pi_i --- actually should be, but to be sure Pi is transported in an extra vector
	// alpha_1(i) = Pi_i*b_i(O_1)

	int startingObs = dev_O_obsSequence_1D[0];

	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		double alpha_1_i = dev_Pi_startProbs_1D[i] * dev_B_obsEmissionProbs_2D[i*V_noOfObsSymbols + startingObs];
		dev_Alpha_trelis_2D[i] = alpha_1_i; // init first row of trelis
	}

	// ------------------------------------------------------------------------------------------------------
	
	for (unsigned int i = 1; i < T_noOfObservations; i++){
		int obs = dev_O_obsSequence_1D[i];

		// call kernel for NxV matrix ops (N is the number of states, T is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.
		//fwKernel << <N_noOfStates, N_noOfStates >> >(dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, obs);
		forwardKernel << <N_noOfStates, N_noOfStates >> >(dev_Alpha_trelis_2D, dev_probs_3D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, obs);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithmCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	// ------------------------------------------------------------------------------------------------------
	// Initialization of the Alpha_trelis
	// ------------------------------------------------------------------------------------------------------
	// a_0i = pi_i --- actually should be, but to be sure Pi is transported in an extra vector
	// alpha_1(i) = Pi_i*b_i(O_1)

	int startingObs = dev_O_obsSequence_1D[0];

	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		double alpha_1_i = dev_Pi_startProbs_1D[i] * dev_B_obsEmissionProbs_2D[i*V_noOfObsSymbols + startingObs];
		dev_Alpha_trelis_2D[i] = alpha_1_i; // init first row of trelis
	}

	// ------------------------------------------------------------------------------------------------------


	// call kernel for NxV matrix ops (N is the number of states, V is the number of observations)
	// Launch a kernel on the GPU with one thread for each element.
//	fwKernel << <N, N >> >(dev_probability, dev_transition, dev_emission, i);
			


	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithm(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = ForwardAlgorithmGPU(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, dev_Alpha_trelis_2D, dev_probs_3D);
		break;
	case ComputationEnvironment::CPU:
		cudaStatus = ForwardAlgorithmCPU(dev_Pi_startProbs_1D, dev_A_stateTransProbs_2D, dev_B_obsEmissionProbs_2D, dev_O_obsSequence_1D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, dev_Alpha_trelis_2D, dev_probs_3D);
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(IntHdl pVector, int numberOfElements)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMalloc((void**)pVector, numberOfElements * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(*pVector);
			*pVector = NULL;
		}
		break;
	case ComputationEnvironment::CPU:
		*pVector = (int *)malloc(numberOfElements * sizeof(int));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(FloatHdl pVector, int numberOfElements)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMalloc((void**)pVector, numberOfElements * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(*pVector);
			*pVector = NULL;
		}
		break;
	case ComputationEnvironment::CPU:
		*pVector = (float *)malloc(numberOfElements * sizeof(float));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(DoubleHdl pVector, int numberOfElements)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMalloc((void**)pVector, numberOfElements * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(*pVector);
			*pVector = NULL;
		}
		break;
	case ComputationEnvironment::CPU:
		*pVector = (double *)malloc(numberOfElements * sizeof(double));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t memcpyVector(IntPtr dst, const IntPtr src, int numberOfElements, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMemcpy(dst, src, numberOfElements * sizeof(int), kind);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		break;
	case ComputationEnvironment::CPU:
		memccpy(dst, src, numberOfElements, sizeof(int));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t memcpyVector(FloatPtr dst, const FloatPtr src, int numberOfElements, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMemcpy(dst, src, numberOfElements * sizeof(float), kind);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		break;
	case ComputationEnvironment::CPU:
		memccpy(dst, src, numberOfElements, sizeof(float));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t memcpyVector(DoublePtr dst, const DoublePtr src, int numberOfElements, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMemcpy(dst, src, numberOfElements * sizeof(double), kind);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		break;
	case ComputationEnvironment::CPU:
		memccpy(dst, src, numberOfElements, sizeof(double));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t deviceFree(void *devPtr)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaFree(devPtr);
		break;
	case ComputationEnvironment::CPU:
		free(devPtr);
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}
