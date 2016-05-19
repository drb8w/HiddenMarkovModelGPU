#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matricies.h"
#include "Observation.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

__global__ void fwKernel(double *p, const double *transition, const double *emission, int obs){

	int ix = blockDim.x*blockIdx.x + threadIdx.x; // i
	int iy = blockDim.y*blockIdx.y + threadIdx.y; // j

	int idx_trans= iy * blockDim.x + ix; // blockDim.x == blockDim.y, cuda_2.pdf s.31
	int idx_emit = ix * blockDim.x + obs;
	int idx_prob = blockDim.x * blockDim.y * obs + blockDim.x * ix + iy;

	double trans = transition[idx_trans];
	double emis = emission[idx_emit];
	p[idx_prob] = trans * emis;


}

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

	cudaStatus = cudaMalloc((void**)&dev_transition, N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_transition);
		cudaFree(dev_emission);
		cudaFree(dev_probability);

		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_emission, V * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_transition);
		cudaFree(dev_emission);
		cudaFree(dev_probability);

		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_transition, matricies->transitionAsArray() , N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_transition);
		cudaFree(dev_emission);
		cudaFree(dev_probability);

		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_emission, matricies->emissionAsArray(), V * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_transition);
		cudaFree(dev_emission);
		cudaFree(dev_probability);

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
		cudaStatus = cudaMalloc((void**)&dev_probability, N * N * T * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(dev_transition);
			cudaFree(dev_emission);
			cudaFree(dev_probability);

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

		for (unsigned int i = 1; i < T; i++){
			int obs = sequence->at(i);

			// call kernel for NxV matrix ops (N is the number of states, V is the number of observations)
			// Launch a kernel on the GPU with one thread for each element.
			fwKernel << <N, N >> >(dev_probability, dev_transition, dev_emission, obs);
			

		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			cudaFree(dev_transition);
			cudaFree(dev_emission);
			cudaFree(dev_probability);

			return cudaStatus;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(host_probability, dev_probability, N * N * T * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			cudaFree(dev_transition);
			cudaFree(dev_emission);
			cudaFree(dev_probability);

			return cudaStatus;
		}

		delete[] host_probability;
		cudaFree(dev_probability);


	}

	cudaFree(dev_transition);
	cudaFree(dev_emission);


	cout << "end\n";

	return 0;
}
