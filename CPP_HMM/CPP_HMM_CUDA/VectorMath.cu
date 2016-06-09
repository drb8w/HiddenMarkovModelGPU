#include "VectorMath.cuh"

//---------------------------------------------------------------------------------------------------------
// CPU - serial implementation
//---------------------------------------------------------------------------------------------------------

__host__ double rowColumnMulMatrixHost(const double *U, const double *V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index - row first
	//---------------------------------------------------------------------------------------------------------

	int idx_u_i0 = index_row_i * dim1_U;
	int idx_v_0j = index_column_j;

	//---------------------------------------------------------------------------------------------------------
	// iterate the arrays
	//---------------------------------------------------------------------------------------------------------
	double result = 0;

	for (int k = 0; k < dim1_U; k++)
	{
		int idx_u_ik = idx_u_i0 + k;
		int idx_v_kj = idx_v_0j + k*dim1_V;

		result += U[idx_u_ik] * V[idx_v_kj];
	}

	return result;
}

__host__ void elementMulMatrixHost(double *w, const double *U, const double *V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{

}

//---------------------------------------------------------------------------------------------------------
// GPU - parallel implementation
// e.g. with reduction
//---------------------------------------------------------------------------------------------------------

__device__ double rowColumnMulMatrixDevice(const double *U, const double *V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{
	//---------------------------------------------------------------------------------------------------------
	// determine index
	//---------------------------------------------------------------------------------------------------------

	// second dimension of matrices needed to stop additional threads from executing

	return 0;
}

__device__ void elementMulMatrixDevice(double *w, const double *U, const double *V, int index_row_i, int index_column_j, int dim1_U, int dim1_V)
{

	// second dimension of matrices needed to stop additional threads from executing
}

