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
	double *dev_gamma_3D = nullptr;
	double *dev_beta_3D = nullptr;
	double *dev_epsilon_3D = nullptr;
	double* dev_likelihood = nullptr;
	double *dev_3D_Trellis_Alpha = nullptr;
	double *dev_3D_Trellis_BF = nullptr;
	double* dev_A_prime_3D = nullptr;
	double* dev_B_prime_3D = nullptr;

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

	if ((cudaStatus = allocateDeviceVector(&dev_gamma_3D, M_noOfObsSequences*N_noOfStates*T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
	}

	if ((cudaStatus = allocateDeviceVector(&dev_likelihood, M_noOfObsSequences)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma_3D);
	}

	if ((cudaStatus = allocateDeviceVector(&dev_3D_Trellis_BF, M_noOfObsSequences * N_noOfStates*T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_likelihood);
	}

	// will be indexed as MxNxT, as HxWxD
	if ((cudaStatus = allocateDeviceVector(&dev_beta_3D, M_noOfObsSequences * N_noOfStates*T_noOfObservations)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_likelihood);
		deviceFree(dev_3D_Trellis_BF);
	}

	// will be indexed as NxNxM, as HxWxD

	if ((cudaStatus = allocateDeviceVector(&dev_A_prime_3D, M_noOfObsSequences* N_noOfStates*N_noOfStates)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_beta_3D);
	}

	// will be indexed as NxVxM, as HxWxD
	if ((cudaStatus = allocateDeviceVector(&dev_B_prime_3D, M_noOfObsSequences* N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_beta_3D);
		deviceFree(dev_A_prime_3D);
	}

	if ((cudaStatus = allocateDeviceVector(&dev_epsilon_3D, M_noOfObsSequences* N_noOfStates*N_noOfStates)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_beta_3D);
		deviceFree(dev_A_prime_3D);
		deviceFree(dev_B_prime_3D);
	}

	if ((cudaStatus = allocateDeviceVector(&dev_O_obsSequences_2D, T_noOfObservations*M_noOfObsSequences)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_beta_3D);
		deviceFree(dev_A_prime_3D);
		deviceFree(dev_B_prime_3D);
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
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_A_stateTransProbs_2D, (double *)host_A_stateTransProbs_2D, N_noOfStates*N_noOfStates, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_B_obsEmissionProbs_2D, (double *)host_B_obsEmissionProbs_2D, N_noOfStates*V_noOfObsSymbols, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	if ((cudaStatus = memcpyVector(dev_O_obsSequences_2D, (unsigned int *)host_O_obsSequences_2D, T_noOfObservations* M_noOfObsSequences, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}


	ForwardAlgorithmSet(host_Pi_startProbs_1D, host_A_stateTransProbs_2D, host_B_obsEmissionProbs_2D, host_O_obsSequences_2D, N_noOfStates, V_noOfObsSymbols, T_noOfObservations, M_noOfObsSequences, host_likelihoods_1D, printToConsole, &dev_3D_Trellis_Alpha, true);

	if ((cudaStatus = memcpyVector(dev_likelihood, (double *)host_likelihoods_1D, M_noOfObsSequences, cudaMemcpyHostToDevice)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		return cudaStatus;
	}

	// --------------------------------------------------------------------------------------------------------
	// init device memory
	// --------------------------------------------------------------------------------------------------------

	for (int m = 0; m < M_noOfObsSequences; m++){
		initArr << <N_noOfStates, N_noOfStates >> >(dev_A_prime_3D,m);
		initArr << <N_noOfStates, V_noOfObsSymbols >> >(dev_B_prime_3D,m);
	}

	initBeta << < M_noOfObsSequences, N_noOfStates >> > (dev_beta_3D, T_noOfObservations);

	//printDeviceMemToScreen(&dev_beta_3D[(T_noOfObservations-1)*N_noOfStates*M_noOfObsSequences], N_noOfStates*M_noOfObsSequences);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

	//AlphaTrellisInitializationGPU << <M_noOfObsSequences, N_noOfStates >> >(dev_3D_Trellis_BF, dev_Pi_startProbs_1D, dev_B_obsEmissionProbs_2D, dev_O_obsSequences_2D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

	/*cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);*/

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// for each obs. sequence do
	for (int t = T_noOfObservations-1; t>=0; t--) {

		if (printToConsole)
			cout << "starting BW alg for obs sequence...\n";

		double* dev_B = nullptr;
		double* dev_W = nullptr;
		double* dev_D = nullptr;

		// --------------------------------------------------------------------------------------------------------
		// Device memory initialization per timestep t
		// --------------------------------------------------------------------------------------------------------


		if ((cudaStatus = allocateDeviceVector(&dev_B, M_noOfObsSequences * N_noOfStates, true)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequences_2D);
			deviceFree(dev_gamma_3D);
			deviceFree(dev_epsilon_3D);
			deviceFree(dev_beta_3D);
			return cudaStatus;
		}

		if ((cudaStatus = allocateDeviceVector(&dev_W, M_noOfObsSequences * N_noOfStates, true)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequences_2D);
			deviceFree(dev_gamma_3D);
			deviceFree(dev_epsilon_3D);
			deviceFree(dev_beta_3D);
			deviceFree(dev_B);
			return cudaStatus;
		}

		if ((cudaStatus = allocateDeviceVector(&dev_D, M_noOfObsSequences * N_noOfStates, true)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequences_2D);
			deviceFree(dev_gamma_3D);
			deviceFree(dev_epsilon_3D);
			deviceFree(dev_beta_3D);
			deviceFree(dev_B);
			deviceFree(dev_W);
			return cudaStatus;
		}


		UpdateGammaGPU << <M_noOfObsSequences, N_noOfStates >> > (dev_gamma_3D, dev_beta_3D, dev_3D_Trellis_Alpha, t, dev_likelihood, T_noOfObservations);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

		if (t > 0){

			// Computes the B matrix in term D = B .* C x A
			ComputeBDevice << < M_noOfObsSequences, N_noOfStates >> >(M_noOfObsSequences, V_noOfObsSymbols, T_noOfObservations, N_noOfStates, dev_O_obsSequences_2D, dev_B_obsEmissionProbs_2D, t, dev_B);
			cudaStatus = cudaDeviceSynchronize();

			// All dimensions are multipe of Warp Sizes
			// Compute W =  B .* C i.e. beta(t,i) * b_it
			pointwiseMatrixMul << <M_noOfObsSequences, N_noOfStates >> >(dev_W, dev_B, &dev_beta_3D[t * M_noOfObsSequences * N_noOfStates]);

			cudaStatus = cudaDeviceSynchronize();

			// Compute D = W x A, D = beta(t,i)*p
			cublasMultiplyDouble(M_noOfObsSequences, N_noOfStates, N_noOfStates, dev_W, dev_A_stateTransProbs_2D, dev_D);


			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();

			updateBeta << <M_noOfObsSequences, N_noOfStates >> >(dev_beta_3D, dev_D, t, T_noOfObservations);

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

			for (int j = 0; j < N_noOfStates; j++){
				UpdateEpsilonGPU << <M_noOfObsSequences, N_noOfStates >> >(dev_epsilon_3D, dev_3D_Trellis_Alpha, t, dev_likelihood, j, dev_D);

				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
			}

		}



		deviceFree(dev_B);
		deviceFree(dev_W);
		deviceFree(dev_D);

	}


	// --------------------------------------------------------------------------------------------------------
	// Estimate Matricies
	// --------------------------------------------------------------------------------------------------------

	double *dev_A_acc_out = nullptr;
	double *epsilon_reduction_grid = nullptr;
	double *gamma_reduction_grid = nullptr;

	if ((cudaStatus = allocateDeviceVector(&dev_A_acc_out, N_noOfStates, true)) != cudaSuccess) {
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&epsilon_reduction_grid, M_noOfObsSequences* N_noOfStates, true)) != cudaSuccess) {
		return cudaStatus;
	}

	if ((cudaStatus = allocateDeviceVector(&gamma_reduction_grid, M_noOfObsSequences* N_noOfStates, true)) != cudaSuccess) {
		return cudaStatus;
	}


	for (int m = 0; m < M_noOfObsSequences; m++){

		for (int i = 0; i < N_noOfStates; i++){

			int smBytes = 64 * sizeof(double);
			int grid = N_noOfStates / 64;
			reduce_1_2D << <grid, 64, smBytes >> >(&dev_epsilon_3D[m*N_noOfStates + i*M_noOfObsSequences*N_noOfStates], dev_A_acc_out, m, M_noOfObsSequences, i, N_noOfStates, epsilon_reduction_grid);

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

		}

		for (int t = 0; t < T_noOfObservations; t++){
			int smBytes = 64 * sizeof(double);
			int grid = N_noOfStates / 64;
			reduce_1_2D << <grid, 64, smBytes >> >(&dev_gamma_3D[m*N_noOfStates + t*M_noOfObsSequences*N_noOfStates], dev_A_acc_out, m, M_noOfObsSequences, t, T_noOfObservations, gamma_reduction_grid);

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

		}

	}

	for (int m = 0; m < M_noOfObsSequences; m++){

		update << <N_noOfStates, N_noOfStates >> >(dev_A_prime_3D, dev_epsilon_3D, epsilon_reduction_grid, m, M_noOfObsSequences);

		if (printToConsole){
			printDeviceMemToScreen(&dev_A_prime_3D[m*N_noOfStates*N_noOfStates], N_noOfStates*N_noOfStates);
		}

		update << <T_noOfObservations, N_noOfStates >> >(dev_B_prime_3D, dev_gamma_3D, gamma_reduction_grid, m, M_noOfObsSequences);

		if (printToConsole){

			printDeviceMemToScreen(&dev_gamma_3D[m*M_noOfObsSequences*N_noOfStates], N_noOfStates*M_noOfObsSequences);
		}


	}





	// --------------------------------------------------------------------------------------------------------


	// --------------------------------------------------------------------------------------------------------
	// device memory cleanup
	// --------------------------------------------------------------------------------------------------------
	deviceFree(dev_Pi_startProbs_1D);
	deviceFree(dev_A_stateTransProbs_2D);
	deviceFree(dev_B_obsEmissionProbs_2D);
	deviceFree(dev_O_obsSequences_2D);
	deviceFree(dev_3D_Trellis_Alpha);
	deviceFree(dev_3D_Trellis_BF);
	deviceFree(dev_gamma_3D);
	deviceFree(dev_beta_3D);
	deviceFree(dev_A_prime_3D);
	deviceFree(dev_B_prime_3D);
	deviceFree(dev_epsilon_3D);
	deviceFree(epsilon_reduction_grid);
	deviceFree(dev_A_acc_out);
	deviceFree(gamma_reduction_grid);

	// --------------------------------------------------------------------------------------------------------

#endif
}

__global__ void UpdateGammaGPU(double* dev_gamma_3D,double *dev_beta_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood, int T_noOfObservations){

	int trellis_idx = t*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;

	int index_2D = blockIdx.x*blockDim.x + threadIdx.x;

	double beta = dev_beta_3D[trellis_idx];
	double trellis = dev_3D_Trellis_Alpha[trellis_idx];
	double val = trellis*beta / dev_likelihood[blockIdx.x];

	dev_gamma_3D[trellis_idx] += val;

}

__global__ void UpdateEpsilonGPU(double* dev_epsilon_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood, int j, double *dev_D){


	int trellis_idx = t*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;
	int epsilon_idx = j*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;

	int index_2D = blockIdx.x*blockDim.x + threadIdx.x;
	double trellis = dev_3D_Trellis_Alpha[trellis_idx];
	double val = trellis * dev_D[index_2D] / dev_likelihood[blockIdx.x];

	dev_epsilon_3D[epsilon_idx] += val;

}

__global__ void initArr(double* arr, int m){

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int offset = m*blockDim.x*gridDim.x;
	double val = 1 / blockDim.x;

	arr[idx] = offset+ val;

}

__global__ void initBeta(double* beta_3D, int T_noOfObservations){

	// 2D index
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int offset = gridDim.x*blockDim.x*(T_noOfObservations-1);

	beta_3D[offset + idx] = 1;

}

__global__ void updateBeta(double* dev_beta_3D, double* dev_D,int t,int T_noOfObservations){

	int idx_2D = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_t_minus_1 = (t - 1)*blockDim.x*gridDim.x + idx_2D;

	dev_beta_3D[idx_t_minus_1] += dev_D[idx_2D];
}

__global__ void update(double* dev_update, double*dev_source, double* reduction_grid,int m, int M){

	int idx_3D = m*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

	//int source_row = m*blockDim.x + blockIdx.x*M*blockDim.x;
	//int col = threadIdx.x;

	int reduction_idx = m*blockDim.x + blockIdx.x;

	double val = dev_source[idx_3D];
	dev_update[idx_3D] = val / reduction_grid[reduction_idx];
	dev_update[idx_3D] = val;

}
