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

__host__ cudaError_t ForwardAlgorithm(vector<unsigned int>* &sequence, int N, double *dev_probability, double *dev_transition, double *dev_emission);
__host__ cudaError_t ForwardAlgorithmGPU(vector<unsigned int>* &sequence, int N, double *dev_probability, double *dev_transition, double *dev_emission);
__host__ cudaError_t ForwardAlgorithmCPU(vector<unsigned int>* &sequence, int N, double *dev_probability, double *dev_transition, double *dev_emission);

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
	double *dev_transition = 0;
	double *dev_emission = 0;
	double *dev_probability = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	Matricies* matricies = new Matricies();
	Observation* observations = new Observation();
	int N = matricies->N;
	int V = matricies->V;

	matricies->loadMatricies(argv[1]);
	observations->loadObservations(argv[1]);

	if ((cudaStatus = allocateDeviceVector(&dev_transition, N)) != cudaSuccess) {
		return cudaStatus;
	}
	
	if ((cudaStatus = allocateDeviceVector(&dev_emission, V)) != cudaSuccess) {
		deviceFree(dev_transition);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	if ((cudaStatus = memcpyVector(dev_transition, matricies->transitionAsArray(), N, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_transition);
		deviceFree(dev_emission);
		deviceFree(dev_probability);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_emission, matricies->emissionAsArray(), V, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_transition);
		deviceFree(dev_emission);
		deviceFree(dev_probability);
		return cudaStatus;
	}

	vector<vector<unsigned int>*>* sequences = &observations->sequences;
	int numberOfObservations = sequences->size();

	// for each obs. sequence do
	for (unsigned int i = 0; i<numberOfObservations; i++) {

		cout << "starting fw alg for obs sequence...\n";

		vector<unsigned int>* sequence = sequences->at(i);
		int T = sequence->size();

		double* host_probability = new double[N * N * T];

		// array to store all probabilities.
		if ((cudaStatus = allocateDeviceVector(&dev_probability, N * N * T)) != cudaSuccess) {
			deviceFree(dev_transition);
			deviceFree(dev_emission);
			return cudaStatus;
		}

		vector<vector<double>*> trelis;
		trelis.resize(T,new vector<double>());
		for (unsigned int i = 0; i < T; i++){
			trelis.at(i)->resize(N, 0);
		}

		int startingObs = sequence->at(0);
		
		//init the trelis
		for (unsigned int i = 0; i < N; i++){
			double initVal = matricies->pi[i] + matricies->emission[i*V + startingObs];
			trelis.at(0)->at(i) = initVal;
		}

		// --------------------------------------------------------------------------------------------------------

		cudaStatus = ForwardAlgorithm(sequence, N, dev_probability, dev_transition, dev_emission);

		// --------------------------------------------------------------------------------------------------------

		if (cudaStatus != cudaSuccess) {
			deviceFree(dev_transition);
			deviceFree(dev_emission);
			deviceFree(dev_probability);
			return cudaStatus;
		}

		// Copy output vector from GPU buffer to host memory.
		if ((cudaStatus = memcpyVector(host_probability, dev_probability, N * N * T, cudaMemcpyDeviceToHost)) != cudaSuccess) {
			deviceFree(dev_transition);
			deviceFree(dev_emission);
			deviceFree(dev_probability);
			return cudaStatus;
		}

		delete[] host_probability;
		deviceFree(dev_probability);

	}

	deviceFree(dev_transition);
	deviceFree(dev_emission);

	cout << "end\n";

	return 0;
}

// ------------------------------------------------------------------------------------------------------

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
// wrapper functions to switch transparent between GPU and CPU calcuation 
// without changing the main algorithms
// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ForwardAlgorithmGPU(vector<unsigned int>* &sequence, int N, double *dev_probability, double *dev_transition, double *dev_emission)
{
	int T = sequence->size();

	for (unsigned int i = 1; i < T; i++){
		int obs = sequence->at(i);

		// call kernel for NxV matrix ops (N is the number of states, V is the number of observations)
		// Launch a kernel on the GPU with one thread for each element.
		fwKernel << <N, N >> >(dev_probability, dev_transition, dev_emission, obs);

	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithmCPU(vector<unsigned int>* &sequence, int N, double *dev_probability, double *dev_transition, double *dev_emission)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	// TODO...
	return cudaStatus;
}

__host__ cudaError_t ForwardAlgorithm(vector<unsigned int>* &sequence, int N, double *dev_probability, double *dev_transition, double *dev_emission)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	int T = sequence->size();

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = ForwardAlgorithmGPU(sequence, N, dev_probability, dev_transition, dev_emission);
		break;
	case ComputationEnvironment::CPU:
		cudaStatus = ForwardAlgorithmCPU(sequence, N, dev_probability, dev_transition, dev_emission);
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
