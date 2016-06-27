#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "VectorMath.cuh"
#include "Trellis.cuh"
#include "forward.cuh"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

__host__ cudaError_t BFAlgorithmSet2D(const double *host_Pi_startProbs_1D, double *host_A_stateTransProbs_2D, double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, int M_noOfObsSequences, double *host_likelihoods_1D, bool printToConsole, std::string fileName, bool profile);

__global__ void initArr(double* arr, int m);
__global__ void initBeta(double* beta_3D, int T_noOfObservations);
__global__ void UpdateGammaGPU(double* dev_gamma_3D, double *dev_beta_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood, int T_noOfObservations);
__global__ void UpdateEpsilonGPU(double* dev_epsilon_3D, double *dev_beta_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood, int j, double *dev_D);
__global__ void EstimateB(double* dev_update, double*dev_epsilon, double* reduction_grid, int m, int M);
__global__ void updateBeta(double* dev_beta_3D, double* dev_D, int t, int T_noOfObservations);
__global__ void compute(double* dev_update, double*dev_source, double* reduction_grid, int m, int M);
__global__ void UpdateEpsilonReductionErrorGPU(double* reduction_grid_error, double *dev_beta_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood);
__global__ void EstimateA(double* dev_update, double*dev_source, double* reduction_grid, double* reduction_grid_error, int m, int M);
__global__ void ColumReduction_Height(double* dev_update, int M);
__global__ void ColumReductionGamma(double* dev_update, int m, int M);
__global__ void ColumReductionGamma_Depth(double* dev_update, int m,int V, int M, double* grid);

void printMatrixForSequence(double* matrix, int m, int row_dim, int col_dim, int depth, std::string fileName, bool isMatrixA);
__global__ void initMatrix(double* matrix_3D, int depth);
void copyMatrix(double* dev_matrix_3D, double* matrix_2D, int row_dim, int col_dim, int depth);
void printMatrix2DToScreen(double* matrix, int m, int row_dim, int col_dim);