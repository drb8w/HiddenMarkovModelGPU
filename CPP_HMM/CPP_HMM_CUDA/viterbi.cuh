#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>

#include "Trellis.cuh"

// ------------------------------------------------------------------------------------------------------
// declarations
// ------------------------------------------------------------------------------------------------------

__host__ __device__ void viterbi2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int i, unsigned int j, unsigned int t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);
__global__ void viterbiKernel2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int T_noOfObservations, unsigned int idx_obs, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
__host__ __device__ void createViterbiIndices2DDevice(unsigned int &idx_a_ji, unsigned int &idx_b_it, unsigned int &idx_p, unsigned int i, unsigned int j, unsigned int t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);
__host__ void createViterbiIndices2DHost(unsigned int &idx_p, unsigned int &idx_alpha_ti, unsigned int &idx_alpha_tm1j, unsigned int i, unsigned int j, unsigned int t, unsigned int dim1_Alpha, unsigned int dim1_P, unsigned int dim2_P);
// ------------------------------------------------------------------------------------------------------
__device__ void createViterbiMatrixDimensions2DDevice(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);
__host__ void createViterbiMatrixDimensions2DHost(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_Alpha, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int N_noOfStates, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
__host__ cudaError_t ForwardAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, double *host_likelihoods_1D);
// ------------------------------------------------------------------------------------------------------
__host__ cudaError_t ViterbiAlgorithm2D(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ViterbiAlgorithm2DGPU(const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ViterbiAlgorithm2DCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
// ------------------------------------------------------------------------------------------------------
//__host__ void GammaTrellisInitialization2D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------