#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>

// ------------------------------------------------------------------------------------------------------
// declarations
// ------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]);

// ------------------------------------------------------------------------------------------------------


__host__ __device__ void createForwardIndices(unsigned int &idx_a_ji, unsigned int &idx_b_it, unsigned int &idx_p, unsigned int &idx_alpha_ti, unsigned int &idx_alpha_tm1j, unsigned int i, unsigned int j, unsigned int t, unsigned int dim1_Alpha, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);
__host__ __device__ void createForwardIndices2D(unsigned int &idx_a_ji, unsigned int &idx_b_it, unsigned int &idx_p, unsigned int i, unsigned int j, unsigned int t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);

__host__ __device__ void forward(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int i, unsigned int j, unsigned int t, unsigned int dim1_Alpha, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);
__host__ __device__ void forward2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int i, unsigned int j, unsigned int t, unsigned int dim1_P, unsigned int dim2_P, unsigned int dim1_A, unsigned int dim1_B);


//__global__ void fwKernel(double *p, const double *transition, const double *emission, int obs);

__device__ void createForwardMatrixDimensionsDevice(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_Alpha, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);
__device__ void createForwardMatrixDimensions2DDevice(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);

__global__ void forwardKernel(double *dev_Alpha_trelis_2D, double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *dev_O_obsSequence_1D, unsigned int T_noOfObservations, unsigned int idx_obs, unsigned int V_noOfObsSymbols);
__global__ void forwardKernel2D(double *dev_probs_3D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int T_noOfObservations, unsigned int idx_obs, unsigned int V_noOfObsSymbols);

// ------------------------------------------------------------------------------------------------------
// version 1.0 
// ------------------------------------------------------------------------------------------------------
//__host__ cudaError_t ForwardAlgorithm(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D, double &likelihood);
//__host__ cudaError_t ForwardAlgorithmGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D, double &likelihood);
//__host__ cudaError_t ForwardAlgorithmCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const int *dev_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *dev_Alpha_trelis_2D, double *dev_probs_3D, double &likelihood);
// ------------------------------------------------------------------------------------------------------
// version 2.0 
// ------------------------------------------------------------------------------------------------------
//__host__ cudaError_t ForwardAlgorithm(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
//__host__ cudaError_t ForwardAlgorithmGPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
//__host__ cudaError_t ForwardAlgorithmCPU(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
// ------------------------------------------------------------------------------------------------------
// version 3.0 
// ------------------------------------------------------------------------------------------------------
__host__ cudaError_t ForwardAlgorithm(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ForwardAlgorithm2D(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ForwardAlgorithmGPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ForwardAlgorithm2DGPU(const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ForwardAlgorithmCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
__host__ cudaError_t ForwardAlgorithm2DCPU(const double *dev_Pi_startProbs_1D, const double *dev_A_stateTransProbs_2D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, double *host_Alpha_trelis_2D, double *host_probs_3D, double &host_likelihood);
// ------------------------------------------------------------------------------------------------------
__host__ cudaError_t CalculateLikelihoodAlphaTrellis2DHost(double &host_likelihood, const double *host_Alpha_trelis_2D, unsigned int N_noOfStates, unsigned int T_noOfObservations);
__host__ void createForwardMatrixDimensionsHost(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_Alpha, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int N_noOfStates, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);
__host__ void createForwardMatrixDimensions2DHost(unsigned int &dim1_A, unsigned int &dim1_B, unsigned int &dim1_P, unsigned int &dim2_P, unsigned int N_noOfStates, unsigned int T_noOfObservations, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
//__host__ cudaError_t ForwardAlgorithmSet(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, int M_noOfObsSequences, double *host_Alpha_trelis_3D, double *host_probs_4D, double *host_likelihoods_1D);
__host__ cudaError_t ForwardAlgorithmSet(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, double *host_likelihoods_1D);
__host__ cudaError_t ForwardAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols, unsigned int T_noOfObservations, unsigned int M_noOfObsSequences, double *host_likelihoods_1D);
// ------------------------------------------------------------------------------------------------------
__host__ void AlphaTrellisInitialization2D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, unsigned int T_noOfObservations, unsigned int N_noOfStates, unsigned int V_noOfObsSymbols);
// ------------------------------------------------------------------------------------------------------
