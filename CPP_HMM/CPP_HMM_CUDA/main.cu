#include "main.cuh"

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

int main(int argc, char* argv[])
{

	cout << "start...\n";

	cudaError_t cudaStatus;
	double *host_Pi_startProbs_1D = nullptr;
	double *host_A_stateTransProbs_2D = nullptr;
	double *host_B_obsEmissionProbs_2D = nullptr;
	unsigned int *host_O_obsSequences_2D = nullptr;
	double *host_likelihoods_1D = nullptr;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceProperties(&prop, 0);

	Matricies* matricies = new Matricies(argv[1]);
	Observation* observations = new Observation();
	int N_noOfStates = matricies->N;
	int V_noOfObsSymbols = matricies->V;

	matricies->loadMatricies(argv[1]);
	observations->loadObservations(argv[1]);

	// --------------------------------------------------------------------------------------------------------
	// access HMM model and data
	// --------------------------------------------------------------------------------------------------------

	host_Pi_startProbs_1D = matricies->piAsArray();
	host_A_stateTransProbs_2D = matricies->transitionAsArray();
	host_B_obsEmissionProbs_2D = matricies->emissionAsArray();
	host_O_obsSequences_2D = observations->observationSequencesAsArray();

	int T_noOfObservations = observations->getTnoOfObservations();
	int M_noOfObsSequences = observations->getMnoOfObsSequences();

	// --------------------------------------------------------------------------------------------------------
	// memory allocation
	// --------------------------------------------------------------------------------------------------------

	host_likelihoods_1D = (double *)calloc(M_noOfObsSequences, sizeof(double));

	// --------------------------------------------------------------------------------------------------------
	// 2D optimization - slow
	// --------------------------------------------------------------------------------------------------------

	cudaStatus = ForwardAlgorithmSet2D(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D);

	// --------------------------------------------------------------------------------------------------------
	// 3D optimization - fast
	// --------------------------------------------------------------------------------------------------------

	//cudaStatus = ForwardAlgorithmSet(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_Alpha_trelis_3D, host_probs_4D, host_likelihoods_1D);
	cudaStatus = ForwardAlgorithmSet(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D);

	// --------------------------------------------------------------------------------------------------------
	// memory cleanup
	// --------------------------------------------------------------------------------------------------------
	// TODO: not nice to clean outside of object
	free(host_O_obsSequences_2D);
	host_O_obsSequences_2D = nullptr;

	free(host_likelihoods_1D);
	host_likelihoods_1D = nullptr;
	// --------------------------------------------------------------------------------------------------------

	cout << "end\n";

	return 0;
}

