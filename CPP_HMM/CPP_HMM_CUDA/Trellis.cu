#include "Trellis.cuh"

#include "Utilities.h"

__host__ __device__ void TrellisScaling2D(double *host_Alpha_trelis_2D, unsigned int T_noOfObservations, unsigned int N_noOfStates, unsigned int idx_t)
{
#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// determine matrix dimensions
	unsigned int dim1_Alpha = N_noOfStates;
	unsigned int dim2_Alpha = T_noOfObservations;

	// determine sum
	double sum_Alpha_t = 0;
	for (unsigned int idx_j = 0; idx_j < dim1_Alpha; idx_j++)
	{
		// determine matrix indices
		unsigned int idx_alpha_tj = idx_t*dim1_Alpha + idx_j;
		sum_Alpha_t += host_Alpha_trelis_2D[idx_alpha_tj];
	}

	if (sum_Alpha_t == 0)
		return;

	double c_t = 1.0 / sum_Alpha_t;

	// rescale alphas
	for (unsigned int idx_j = 0; idx_j < dim1_Alpha; idx_j++)
	{
		// determine matrix indices
		unsigned int idx_alpha_tj = idx_t*dim1_Alpha + idx_j;
		host_Alpha_trelis_2D[idx_alpha_tj] *= c_t;
	}

#endif
}

__host__ __device__ void TrellisInitialization2D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols)
{

	// ------------------------------------------------------------------------------------------------------
	// Initialization of the Alpha_trelis
	// in the paper the initialization of the trellis is done differently, in code actually it is an initialization from the priors
	// ------------------------------------------------------------------------------------------------------

	// a_0i = pi_i --- actually data should be set up like this, but to be sure Pi is transported in an extra vector
	// alpha_0(i) = Pi_i*b_i(O_0)

	int obs_start = host_O_obsSequence_1D[0];
	// TODO: similar to the following
	//Observation observation;
	//idx_obs_T = observation.getObservationSymbolIndex(obs_start);
	// HACK: symbol id is same as index
	int idx_obs_start = obs_start;

#ifdef COL_MAJ_ORD_MAT_ROW_FIRST_INDEX

	int dim1_B = N_noOfStates;
	int dim1_Alpha = T_noOfObservations;
	// init first row of trellis
	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		int idx_b_i_idxOs = i + idx_obs_start * dim1_B;
		int idx_alpha_0i = i*dim1_Alpha;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	//for (i = 0; i < nstates; i++) {
	//	alpha[0][i] = prior[i] + obvs[IDX(i, data[0], nobvs)];
	//}

	// init first row of trellis
	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		int idx_b_i_idxOs = i*V_noOfObsSymbols + idx_obs_start;
		int idx_alpha_0i = i;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif
}

//---------------------------------------------------------------------------------------------------------

__host__ void AlphaTrellisSliceInitialization2D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int N_noOfStates, int V_noOfObsSymbols, int M_noOfSequences, int idx_m)
{

	// ------------------------------------------------------------------------------------------------------
	// Initialization of the Alpha_trelis
	// in the paper the initialization of the trellis is done differently, in code actually it is an initialization from the priors
	// ------------------------------------------------------------------------------------------------------

	// a_0i = pi_i --- actually data should be set up like this, but to be sure Pi is transported in an extra vector
	// alpha_0(i) = Pi_i*b_i(O_0)

	int obs_start = host_O_obsSequence_1D[0];
	// TODO: similar to the following
	//Observation observation;
	//idx_obs_T = observation.getObservationSymbolIndex(obs_start);
	// HACK: symbol id is same as index
	int idx_obs_start = obs_start;

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX

	// init first row of trellis
	for (unsigned int i = 0; i < N_noOfStates; i++)
	{
		int idx_b_i_idxOs = i*V_noOfObsSymbols + idx_obs_start;
		int idx_alpha_0i = idx_m*N_noOfStates + i;
		int idx_pi_i = i;

		double alpha_0_i = host_Pi_startProbs_1D[idx_pi_i] * host_B_obsEmissionProbs_2D[idx_b_i_idxOs];
		host_Alpha_trelis_2D[idx_alpha_0i] = alpha_0_i;
	}

#endif
}

__global__ void AlphaTrellisInitializationGPU(double *dev_3D_Trellis, const double *dev_Pi_startProbs_1D, const double *dev_B_obsEmissionProbs_2D, const unsigned int *dev_O_obsSequences_2D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols){

	int obs_index = blockIdx.x * T_noOfObservations;
	int obs_start = dev_O_obsSequences_2D[obs_index];
	int idx_b_i_idxOs = threadIdx.x*V_noOfObsSymbols + obs_start;
	int idx_alpha_0i = blockIdx.x * N_noOfStates + threadIdx.x;
	int idx_pi_i = threadIdx.x;

	double alpha_0_i = dev_Pi_startProbs_1D[idx_pi_i] * dev_B_obsEmissionProbs_2D[idx_b_i_idxOs];
	dev_3D_Trellis[idx_alpha_0i] = alpha_0_i;


}

