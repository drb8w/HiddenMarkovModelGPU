#include "BF.cuh"

#include "MemoryManagement.cuh"

#include "Utilities.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

extern ComputationEnvironment glob_Env;

__host__ cudaError_t BFAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, int M_noOfObsSequences, double *host_likelihoods_1D, bool printToConsole){
	if (printToConsole)
		cout << "starting BW alg for obs sequence...\n";

	glob_Env = ComputationEnvironment::GPU;

	double *dev_Pi_startProbs_1D = nullptr;
	double *dev_A_stateTransProbs_2D = nullptr;
	double *dev_B_obsEmissionProbs_2D = nullptr;
	unsigned int *dev_O_obsSequences_2D = nullptr;
	double *dev_Beta = nullptr;
	double *dev_gamma = nullptr;
	double *dev_epsilon = nullptr;
	double *dev_3D_Trellis = nullptr; 

	cudaError_t cudaStatus = cudaSuccess;

	// --------------------------------------------------------------------------------------------------------
	// device memory allocation
	// --------------------------------------------------------------------------------------------------------

	if ((cudaStatus = allocateDeviceVector(&dev_Pi_startProbs_1D, N_noOfStates,true)) != cudaSuccess) {
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

	if ((cudaStatus = allocateDeviceVector(&dev_gamma, N_noOfStates*T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&dev_epsilon, N_noOfStates*N_noOfStates)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma);
		return cudaStatus;
	}

	//if ((cudaStatus = allocateDeviceVector(&dev_3D_Trellis, M_noOfObsSequences * N_noOfStates*T_noOfObservations)) != cudaSuccess) {
	//	deviceFree(dev_Pi_startProbs_1D);
	//	deviceFree(dev_A_stateTransProbs_2D);
	//	deviceFree(dev_B_obsEmissionProbs_2D);
	//	deviceFree(dev_gamma);
	//	deviceFree(dev_epsilon);
	//}

	if ((cudaStatus = allocateDeviceVector(&dev_Beta, N_noOfStates*T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma);
		deviceFree(dev_epsilon);
		//deviceFree(dev_3D_Trellis);
	}

	if ((cudaStatus = allocateDeviceVector(&dev_O_obsSequences_2D, T_noOfObservations* M_noOfObsSequences)) != cudaSuccess) {

		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma);
		deviceFree(dev_epsilon);
		//deviceFree(dev_3D_Trellis);
		deviceFree(dev_Beta);
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
		deviceFree(dev_gamma);
		deviceFree(dev_epsilon);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_Beta);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_A_stateTransProbs_2D, (double *)host_A_stateTransProbs_2D, N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma);
		deviceFree(dev_epsilon);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_Beta);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_B_obsEmissionProbs_2D, (double *)host_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma);
		deviceFree(dev_epsilon);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_Beta);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_O_obsSequences_2D, (unsigned int *)host_O_obsSequences_2D, T_noOfObservations* M_noOfObsSequences, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma);
		deviceFree(dev_epsilon);
		deviceFree(dev_3D_Trellis);
		deviceFree(dev_Beta);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// TODO: the M different observation sequences could be computed in parallel

	// for each obs. sequence do
	for (unsigned int i = 0; i<M_noOfObsSequences; i++) {

		if (printToConsole)
			cout << "starting BW alg for obs sequence...\n";

		double *B_init_host = (double *)calloc(N_noOfStates, sizeof(double));

		for (int i = 0; i < N_noOfStates; i++){
			B_init_host[i] = 1;
		}

		memcpyVector(&dev_Beta[(T_noOfObservations-1)*N_noOfStates], B_init_host, N_noOfStates, cudaMemcpyHostToDevice);

		ForwardAlgorithmSet(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D, false,dev_3D_Trellis,true);

		for (int i = 0; i < M_noOfObsSequences; i++)
		{
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
			// Device memory initialization
			// --------------------------------------------------------------------------------------------------------

			double likeihood = host_likelihoods_1D[i];

			AlphaTrellisInitializationGPU << <M_noOfObsSequences, N_noOfStates >> >(dev_3D_Trellis, dev_Pi_startProbs_1D, dev_B_obsEmissionProbs_2D, dev_O_obsSequences_2D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

			SetGammaEpsilonGPU << <T_noOfObservations, N_noOfStates >> > (dev_gamma, dev_epsilon, dev_Beta, dev_3D_Trellis, i, likeihood, M_noOfObsSequences);

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		}



	}

	// --------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// --------------------------------------------------------------------------------------------------------
	deviceFree(dev_Pi_startProbs_1D);
	deviceFree(dev_A_stateTransProbs_2D);
	deviceFree(dev_B_obsEmissionProbs_2D);
	deviceFree(dev_gamma);
	deviceFree(dev_epsilon);
	deviceFree(dev_3D_Trellis);
	deviceFree(dev_Beta);
	deviceFree(dev_O_obsSequences_2D);

	// --------------------------------------------------------------------------------------------------------

#endif
}

__global__ void SetGammaEpsilonGPU(double* dev_gamma, double* dev_epsilon, double *dev_Beta, double * dev_3D_Trellis, int m, double likelihood, int M_noOfObsSequences){

	// m*N
	int m_offset = m*blockDim.x;
	int trellis_offset = blockIdx.x % gridDim.x;
	int trellis_row = trellis_row*M_noOfObsSequences*blockDim.x;
	int trellis_access = gridDim.x;

	int trellis_idx = m_offset + trellis_row + trellis_access;

	int index_2D = blockIdx.x*blockDim.x + threadIdx.x;

	double beta = dev_Beta[index_2D];
	double gamma = dev_gamma[index_2D];
	double trellis = dev_3D_Trellis[trellis_idx];

	dev_gamma[index_2D] = dev_gamma[index_2D] + trellis*beta / likelihood;

	// TODO reduction(?) for epsilon - not sure about what the paper



}
