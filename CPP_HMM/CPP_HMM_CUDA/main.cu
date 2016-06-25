#include "main.cuh"

#include "forward.cuh"
#include "viterbi.cuh"
#include "BF.cuh"

#include "MemoryManagement.cuh"

#include "Matricies.h"
#include "Observation.h"

#include "Utilities.h"

using namespace std;

extern unsigned int glob_blocksize;
extern ComputationEnvironment glob_Env;

int main(int argc, char* argv[])
{

	cout << "start...\n";

	cudaError_t cudaStatus;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr; // used for event timing

	clock_t start_time;
	clock_t end_time;

	double *host_Pi_startProbs_1D = nullptr;
	double *host_A_stateTransProbs_2D = nullptr;
	double *host_B_obsEmissionProbs_2D = nullptr;
	unsigned int *host_O_obsSequences_2D = nullptr;
	double *host_likelihoods_1D = nullptr;
	unsigned int* host_likeliestStateSequence_2D = nullptr;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceProperties(&prop, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceProperties failed! Cannot get maxThreadsPerBlock.");
		return cudaStatus;
	}
	glob_blocksize = prop.maxThreadsPerBlock;

	Matricies* matricies = new Matricies(argv[1]);
	Observation* observations = new Observation();
	int N_noOfStates = matricies->N;
	int V_noOfObsSymbols = matricies->V;

	matricies->loadMatricies(argv[1]);
	observations->loadObservations(argv[1]);

	initBenchmark(&start,&stop);

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
	host_likeliestStateSequence_2D = (unsigned int *)calloc(M_noOfObsSequences*T_noOfObservations, sizeof(unsigned int));

	// --------------------------------------------------------------------------------------------------------
	// 2D optimization - slow
	// --------------------------------------------------------------------------------------------------------

	glob_Env = ComputationEnvironment::GPU;

	startBenchmark(start, &start_time);

	for (int i = 0; i < ITERATIONS; i++)
	{
		cudaStatus = ForwardAlgorithmSet2D(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D,false);
	}

	stopBenchmark("FWD 2D GPU",start,stop,&start_time,&end_time);

	glob_Env = ComputationEnvironment::CPU;

	startBenchmark(start, &start_time);

	for (int i = 0; i < ITERATIONS; i++)
	{
		cudaStatus = ForwardAlgorithmSet2D(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D, false);
	}

	stopBenchmark("FWD 2D CPU", start, stop, &start_time, &end_time);

	glob_Env = ComputationEnvironment::GPU;

	startBenchmark(start, &start_time);

	for (int i = 0; i < ITERATIONS; i++)
	{
		cudaStatus = ViterbiAlgorithmSet2D(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likeliestStateSequence_2D, true);
	}

	stopBenchmark("Viterbi", start, stop, &start_time, &end_time);

	// --------------------------------------------------------------------------------------------------------
	// 3D optimization - fast
	// --------------------------------------------------------------------------------------------------------

	startBenchmark(start, &start_time);

	for (int i = 0; i < ITERATIONS; i++)
	{
		cudaStatus = ForwardAlgorithmSet(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D,false,nullptr,false);
	}

	stopBenchmark("FWD 3D", start, stop, &start_time, &end_time);

	glob_Env = ComputationEnvironment::ALL;

	startBenchmark(start, &start_time);


	for (int i = 0; i < ITERATIONS; i++)
	{
		cudaStatus = BFAlgorithmSet2D(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D, true, argv[1]);
	}

	stopBenchmark("Baum Welch", start, stop, &start_time, &end_time);

	// --------------------------------------------------------------------------------------------------------
	// memory cleanup
	// --------------------------------------------------------------------------------------------------------
	// TODO: not nice to clean outside of object
	free(host_O_obsSequences_2D);
	host_O_obsSequences_2D = nullptr;

	free(host_likelihoods_1D);
	host_likelihoods_1D = nullptr;

	free(host_likeliestStateSequence_2D);
	host_likeliestStateSequence_2D = nullptr;
	// --------------------------------------------------------------------------------------------------------

	cout << "end\n";

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	return 0;
}

