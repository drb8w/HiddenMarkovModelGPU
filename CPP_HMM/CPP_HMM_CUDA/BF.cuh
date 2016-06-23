#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "VectorMath.cuh"
#include "Trellis.cuh"
#include "forward.cuh"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>

__host__ cudaError_t BFAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, int M_noOfObsSequences, double *host_likelihoods_1D, bool printToConsole);

__global__ void initArr(double* arr);
__global__ void SetGammaEpsilonGPU(double* dev_gamma, double* dev_epsilon, double *dev_Beta, double * dev_3D_Trellis, int m, double likelihood, int M_noOfObsSequences, double* g_idata, double* g_odata);