#include "BF.cuh"

#include "MemoryManagement.cuh"

#include "Utilities.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

extern ComputationEnvironment glob_Env;

__host__ cudaError_t BFAlgorithmSet2D(const double *host_Pi_startProbs_1D, const double *host_A_stateTransProbs_2D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequences_2D, int N_noOfStates, int V_noOfObsSymbols, int T_noOfObservations, int M_noOfObsSequences, double *host_likelihoods_1D, bool printToConsole, string fileName, bool profile){
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
	double* dev_A_prime_3D = nullptr;
	double* dev_B_prime_3D = nullptr;
	double *epsilon_reduction_grid = nullptr;

	cudaError_t cudaStatus = cudaSuccess;

	// --------------------------------------------------------------------------------------------------------
	// device memory allocation
	// --------------------------------------------------------------------------------------------------------

	if ((cudaStatus = allocateDeviceVector(&dev_Pi_startProbs_1D, N_noOfStates, true)) != cudaSuccess) {
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

	if ((cudaStatus = allocateDeviceVector(&dev_gamma_3D, M_noOfObsSequences*N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
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


	// will be indexed as MxNxV, as HxWxD
	if ((cudaStatus = allocateDeviceVector(&dev_beta_3D, M_noOfObsSequences * N_noOfStates*V_noOfObsSymbols)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_likelihood);
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

	// will be indexed as MxNxV, as HxWxD
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

	if ((cudaStatus = allocateDeviceVector(&epsilon_reduction_grid, N_noOfStates, true)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_beta_3D);
		deviceFree(dev_A_prime_3D);
		deviceFree(dev_B_prime_3D);
		return cudaStatus;
	}

	double *gamma_reduction_grid = nullptr;


	if ((cudaStatus = allocateDeviceVector(&gamma_reduction_grid, N_noOfStates, true)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_beta_3D);
		deviceFree(dev_A_prime_3D);
		deviceFree(dev_B_prime_3D);
		return cudaStatus;
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
		initArr << <N_noOfStates, N_noOfStates >> >(dev_A_prime_3D, m);
		initArr << <V_noOfObsSymbols, N_noOfStates >> >(dev_B_prime_3D, m);
	}

	initBeta << < M_noOfObsSequences, N_noOfStates >> > (dev_beta_3D, V_noOfObsSymbols);

	//printDeviceMemToScreen(&dev_beta_3D[(T_noOfObservations-1)*N_noOfStates*M_noOfObsSequences], N_noOfStates*M_noOfObsSequences);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

	//AlphaTrellisInitializationGPU << <M_noOfObsSequences, N_noOfStates >> >(dev_3D_Trellis_BF, dev_Pi_startProbs_1D, dev_B_obsEmissionProbs_2D, dev_O_obsSequences_2D, T_noOfObservations, N_noOfStates, V_noOfObsSymbols);

	/*cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);*/

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	double* epsilon_reduction_grid_error = nullptr;

	if ((cudaStatus = allocateDeviceVector(&epsilon_reduction_grid_error, M_noOfObsSequences* N_noOfStates, true)) != cudaSuccess) {
		deviceFree(dev_Pi_startProbs_1D);
		deviceFree(dev_A_stateTransProbs_2D);
		deviceFree(dev_B_obsEmissionProbs_2D);
		deviceFree(dev_O_obsSequences_2D);
		deviceFree(dev_gamma_3D);
		deviceFree(dev_epsilon_3D);
		deviceFree(dev_beta_3D);
		return cudaStatus;
	}

	// for each obs. sequence do
	for (int t = T_noOfObservations - 1; t >= 0; t--) {

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





		UpdateGammaGPU << <M_noOfObsSequences, N_noOfStates >> > (dev_gamma_3D, dev_beta_3D, dev_3D_Trellis_Alpha, t, dev_likelihood, V_noOfObsSymbols);

		cudaStatus = cudaDeviceSynchronize();

		UpdateEpsilonReductionErrorGPU << <M_noOfObsSequences, N_noOfStates >> >(epsilon_reduction_grid_error, dev_beta_3D, dev_3D_Trellis_Alpha, t, dev_likelihood);

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

			updateBeta << <M_noOfObsSequences, N_noOfStates >> >(dev_beta_3D, dev_D, t, V_noOfObsSymbols);

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

			for (int j = 0; j < N_noOfStates; j++){
				UpdateEpsilonGPU << <M_noOfObsSequences, N_noOfStates >> >(dev_epsilon_3D, dev_beta_3D, dev_3D_Trellis_Alpha, t, dev_likelihood, j, dev_D);

				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
			}

		}



		deviceFree(dev_B);
		deviceFree(dev_W);
		deviceFree(dev_D);


	}


	ColumReduction << <1, N_noOfStates >> >(epsilon_reduction_grid_error, M_noOfObsSequences);


	// --------------------------------------------------------------------------------------------------------
	// Estimate Matricies - 	// sum up all values and reductions then divide
	// --------------------------------------------------------------------------------------------------------





	for (int m = 0; m < M_noOfObsSequences; m++){

		ColumReductionGamma << <V_noOfObsSymbols,N_noOfStates >> >(dev_gamma_3D, m, M_noOfObsSequences);



	}

	ColumReductionGamma_Depth << <M_noOfObsSequences, N_noOfStates >> >(dev_gamma_3D, 0, V_noOfObsSymbols, gamma_reduction_grid);

	//for (int t = 0; t < N_noOfStates; t++){
	//	double *dev_A_acc_out = nullptr;

	//	if ((cudaStatus = allocateDeviceVector(&dev_A_acc_out, V_noOfObsSymbols, true)) != cudaSuccess) {
	//		return cudaStatus;
	//	}

	//	int smBytes = 64 * sizeof(double);
	//	int grid = V_noOfObsSymbols / 64;
	//	reduce_1_2D << <grid, 64, smBytes >> >(&dev_gamma_3D[0*V_noOfObsSymbols + t*M_noOfObsSequences*V_noOfObsSymbols], dev_A_acc_out, 0, M_noOfObsSequences, t, N_noOfStates, gamma_reduction_grid);

	//	cudaStatus = cudaDeviceSynchronize();
	//	if (cudaStatus != cudaSuccess)
	//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

	//	deviceFree(dev_A_acc_out);

	//}

	ColumReduction << <N_noOfStates, N_noOfStates >> >(dev_epsilon_3D, M_noOfObsSequences);

	updateEpsilon << <N_noOfStates, N_noOfStates >> >(dev_A_prime_3D, dev_epsilon_3D, epsilon_reduction_grid, epsilon_reduction_grid_error, 0, M_noOfObsSequences);

	update << <V_noOfObsSymbols,N_noOfStates >> >(dev_B_prime_3D, dev_gamma_3D, gamma_reduction_grid, 0, M_noOfObsSequences);

	//printDeviceMemToScreen(&dev_B_prime_3D[4 * M_noOfObsSequences*N_noOfStates], M_noOfObsSequences* N_noOfStates);


	if (!profile){

		double *A_host = (double*)calloc(M_noOfObsSequences* N_noOfStates*N_noOfStates, sizeof(double));
		double *B_host = (double*)calloc(M_noOfObsSequences* N_noOfStates*V_noOfObsSymbols, sizeof(double));

		if ((cudaStatus = memcpyVector(A_host, (double *)dev_A_prime_3D, M_noOfObsSequences* N_noOfStates*N_noOfStates, cudaMemcpyDeviceToHost)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequences_2D);
			deviceFree(dev_3D_Trellis_Alpha);
			deviceFree(dev_gamma_3D);
			deviceFree(dev_beta_3D);
			deviceFree(dev_A_prime_3D);
			deviceFree(dev_B_prime_3D);
			deviceFree(dev_epsilon_3D);
			deviceFree(epsilon_reduction_grid);
			deviceFree(gamma_reduction_grid);
			return cudaStatus;
		}

		if ((cudaStatus = memcpyVector(B_host, (double *)dev_B_prime_3D, V_noOfObsSymbols* N_noOfStates*M_noOfObsSequences, cudaMemcpyDeviceToHost)) != cudaSuccess) {
			deviceFree(dev_Pi_startProbs_1D);
			deviceFree(dev_A_stateTransProbs_2D);
			deviceFree(dev_B_obsEmissionProbs_2D);
			deviceFree(dev_O_obsSequences_2D);
			deviceFree(dev_3D_Trellis_Alpha);
			deviceFree(dev_gamma_3D);
			deviceFree(dev_beta_3D);
			deviceFree(dev_A_prime_3D);
			deviceFree(dev_B_prime_3D);
			deviceFree(dev_epsilon_3D);
			deviceFree(epsilon_reduction_grid);
			deviceFree(gamma_reduction_grid);
			return cudaStatus;
		}


		cout << "Matrix A" << "\n";

		for (int m = 0; m < 1; m++){

			printMatrixForSequence(A_host, m, N_noOfStates, N_noOfStates, M_noOfObsSequences, fileName, true);

		}

		cout << "Matrix B" << "\n";

		for (int m = 0; m < 1; m++){

			printMatrixForSequence(B_host, m, N_noOfStates, V_noOfObsSymbols, M_noOfObsSequences, fileName, false);

		}

		free(A_host);
		free(B_host);


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
	deviceFree(dev_gamma_3D);
	deviceFree(dev_beta_3D);
	deviceFree(dev_A_prime_3D);
	deviceFree(dev_B_prime_3D);
	deviceFree(dev_epsilon_3D);
	deviceFree(epsilon_reduction_grid);
	deviceFree(gamma_reduction_grid);
	deviceFree(epsilon_reduction_grid_error);

	// --------------------------------------------------------------------------------------------------------

#endif
}

void printMatrixForSequence(double* matrix, int m, int row_dim, int col_dim, int depth, string fileName, bool isMatrixA){


	string rowS = "";
	string colS = "";
	string NAME = "";
	fstream fh;

	if (isMatrixA){

		rowS = "w";
		colS = "w";

		/* init files*/
		NAME = fileName + to_string(m) + ".transLearned";

	}

	else{

		rowS = "w";
		colS = "o";

		NAME = fileName + to_string(m) + ".emitLearned";

	}

	fh.open(NAME.c_str(), fstream::out | fstream::in | fstream::trunc);


	for (int r = 0; r < row_dim; r++){

		for (int c = 0; c < col_dim; c++){

			int idx_3D = c*row_dim*depth + r;

			fh << rowS + to_string(r + 1) << " " << colS + to_string(c + 1) << " " << matrix[idx_3D] << "\n";
		}

	}

	fh.close();

}

__global__ void UpdateGammaGPU(double* dev_gamma_3D, double *dev_beta_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood, int T_noOfObservations){

	int trellis_idx = t*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;

	int index_2D = blockIdx.x*blockDim.x + threadIdx.x;

	double beta = dev_beta_3D[trellis_idx];
	double trellis = dev_3D_Trellis_Alpha[trellis_idx];
	double val = trellis*beta / dev_likelihood[blockIdx.x];

	dev_gamma_3D[trellis_idx] += val;

}

__global__ void UpdateEpsilonGPU(double* dev_epsilon_3D, double *dev_beta_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood, int j, double *dev_D){


	int trellis_idx = t*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;
	int epsilon_idx = j*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;

	int index_2D = blockIdx.x*blockDim.x + threadIdx.x;
	double trellis = dev_3D_Trellis_Alpha[trellis_idx];
	double beta = dev_beta_3D[trellis_idx];
	double val = trellis * dev_D[index_2D] / dev_likelihood[blockIdx.x];
	double factor = (trellis*beta) / (dev_likelihood[blockIdx.x] * blockDim.x);

	dev_epsilon_3D[epsilon_idx] += val;

}

__global__ void UpdateEpsilonReductionErrorGPU(double* reduction_grid_error, double *dev_beta_3D, double * dev_3D_Trellis_Alpha, int t, double* dev_likelihood){


	int trellis_idx = t*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;
	int reduction_row = blockIdx.x*blockDim.x;
	int reduction_idx = blockIdx.x*blockDim.x + threadIdx.x;

	int index_2D = blockIdx.x*blockDim.x + threadIdx.x;
	double trellis = dev_3D_Trellis_Alpha[trellis_idx];
	double beta = dev_beta_3D[trellis_idx];
	double factor = (trellis*beta) / (dev_likelihood[blockIdx.x]);


	reduction_grid_error[reduction_idx] = factor;

	__syncthreads();

	if (threadIdx.x == 0){

		for (int i = 1; i < blockDim.x; i++){
			reduction_grid_error[reduction_row] += reduction_grid_error[reduction_row + i];
		}

		for (int i = 1; i < blockDim.x; i++){
			reduction_grid_error[reduction_row + i] += reduction_grid_error[reduction_row];
		}

	}





}

__global__ void initArr(double* arr, int m){

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int offset = m*blockDim.x*gridDim.x;
	double val = 1 / blockDim.x;

	arr[idx] = offset + val;

}

__global__ void initBeta(double* beta_3D, int T_noOfObservations){

	// 2D index
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int offset = gridDim.x*blockDim.x*(T_noOfObservations - 1);

	beta_3D[offset + idx] = 1;

}

__global__ void updateBeta(double* dev_beta_3D, double* dev_D, int t, int T_noOfObservations){

	int idx_2D = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_t_minus_1 = (t - 1)*blockDim.x*gridDim.x + idx_2D;


	dev_beta_3D[idx_t_minus_1] += dev_D[idx_2D];
}

__global__ void update(double* dev_update, double*dev_source, double* reduction_grid, int m, int M){

	int idx_3D = m * blockDim.x + blockIdx.x*blockDim.x*M + threadIdx.x;
	int idx_top = blockIdx.x*blockDim.x*M + threadIdx.x;

	int reduction_idx = threadIdx.x;

	double val = dev_source[idx_top];
	dev_update[idx_top] = val / reduction_grid[reduction_idx];

}

__global__ void updateEpsilon(double* dev_update, double*dev_source, double* reduction_grid, double* reduction_grid_error, int m, int M){

	int idx_3D = m*blockDim.x + blockIdx.x*blockDim.x*M + threadIdx.x;
	int idx_top = blockIdx.x*blockDim.x*M + threadIdx.x;

	int reduction_idx = m*blockDim.x + blockIdx.x;
	int reduction_idx_1D = blockIdx.x;

	double val = dev_source[idx_top];
	dev_update[idx_top] = val / (reduction_grid_error[reduction_idx_1D]);


}

__global__ void ColumReduction(double* dev_update, int M){

	int start = threadIdx.x + blockIdx.x*M*gridDim.x;


	for (int i = 1; i < M; i++){

		int idx = start + i*blockDim.x;

		dev_update[start] += dev_update[idx];

	}



}

__global__ void ColumReductionGamma(double* dev_update, int m, int M){

	int start = threadIdx.x + m*blockDim.x*M;


	for (int i = 0; i < M; i++){

		int idx = start + i*blockDim.x;

		dev_update[start] += dev_update[idx];

	}



}

__global__ void ColumReductionGamma_Depth(double* dev_update, int m, int M, double* grid){

	int start = threadIdx.x;


	for (int i = 1; i < M; i++){

		int idx = start + i*blockDim.x*M;

		grid[threadIdx.x] += dev_update[idx];

	}



}

__global__ void compute(double* dev_update, double*dev_source, double* reduction_grid, int m, int M){

	int idx_top = blockIdx.x*blockDim.x*M + threadIdx.x;

	int reduction_idx = blockIdx.x;
	//dev_update[idx_3D] = val/reduction_grid[reduction_idx];
	dev_update[idx_top] = dev_update[idx_top] / reduction_grid[reduction_idx];

}

