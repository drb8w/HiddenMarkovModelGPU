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
__host__ __device__ void viterbi1D(double *dev_Alpha_trelis_2D, double *dev_Gamma_trellis_backtrace_2D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_Alpha, unsigned int dim1_A, unsigned int dim1_B);
__global__ void viterbiKernel1D(double *dev_Alpha_trelis_TNM_3D, double *dev_Gamma_trellis_backtrace_TNM_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int T_noOfObservations, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
__host__ __device__ void viterbi2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);
__global__ void viterbiKernel2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int T_noOfObservations, unsigned int idx_t, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
__host__ __device__ void createViterbiIndices1D(unsigned int &idx_a_ji, unsigned int &idx_b_it, unsigned int &idx_alpha_ti, unsigned int &idx_alpha_tm1j,
	unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_A, unsigned int dim1_B, unsigned int dim1_Alpha);
// ------------------------------------------------------------------------------------------------------
__host__ __device__ void createViterbiIndices2DDevice(unsigned int &idx_a_ji, unsigned int &idx_b_it, unsigned int &idx_p, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);
__host__ void createViterbiIndices2DHost(unsigned int &idx_p, unsigned int &idx_alpha_ti, unsigned int &idx_alpha_tm1j, unsigned int idx_i, unsigned int idx_j, unsigned int idx_t, unsigned int dim1_Alpha, unsigned int dim1_P, unsigned int dim2_P);
// ------------------------------------------------------------------------------------------------------
__device__ void createViterbiMatrixDimensions1D(unsigned int &dim1_Alpha, unsigned int &dim2_Alpha, unsigned int &dim1_A, unsigned int &dim1_B, unsigned int T_noOfObservations, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
__device__ void createViterbiMatrixDimensions2DDevice(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);
__host__ void createViterbiMatrixDimensions2DHost(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_Alpha, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int N_noOfStates, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
//__host__ cudaError_t ViterbiAlgorithm1D(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D);
__host__ cudaError_t ViterbiAlgorithm1DGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D);
__host__ cudaError_t ViterbiAlgorithm1DCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D);
// ------------------------------------------------------------------------------------------------------
__host__ cudaError_t ViterbiAlgorithm2D(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D);
__host__ cudaError_t ViterbiAlgorithm2DGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D);
__host__ cudaError_t ViterbiAlgorithm2DCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_Gamma_trellis_backtrace_2D, double *host_probs_3D, unsigned int *host_likeliestStateIndexSequence_1D);
// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ViterbiAlgorithmSet1DGPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D,
	unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, unsigned int *host_likeliestStateIndexSequence_2D);

__host__ cudaError_t ViterbiAlgorithmSet1DCPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, 
	unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, unsigned int *host_likeliestStateIndexSequence_2D);

// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t ViterbiAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, unsigned int *host_likeliestStateIndexSequence_2D, bool printToConsole);
// ------------------------------------------------------------------------------------------------------
//__host__ void GammaTrellisInitialization2D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
