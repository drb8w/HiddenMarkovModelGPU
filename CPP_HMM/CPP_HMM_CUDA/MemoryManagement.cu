#include "MemoryManagement.cuh"

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>

ComputationEnvironment glob_Env = ComputationEnvironment::GPU;

__host__ cudaError_t allocateDeviceVector(IntHdl pVector, int numberOfElements)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMalloc((void**)pVector, numberOfElements * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(*pVector);
			*pVector = NULL;
		}
		break;
	case ComputationEnvironment::CPU:
		*pVector = (int *)malloc(numberOfElements * sizeof(int));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(FloatHdl pVector, int numberOfElements)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMalloc((void**)pVector, numberOfElements * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(*pVector);
			*pVector = NULL;
		}
		break;
	case ComputationEnvironment::CPU:
		*pVector = (float *)malloc(numberOfElements * sizeof(float));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(DoubleHdl pVector, int numberOfElements)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMalloc((void**)pVector, numberOfElements * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(*pVector);
			*pVector = NULL;
		}
		break;
	case ComputationEnvironment::CPU:
		*pVector = (double *)malloc(numberOfElements * sizeof(double));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t memcpyVector(IntPtr dst, const IntPtr src, int numberOfElements, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMemcpy(dst, src, numberOfElements * sizeof(int), kind);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		break;
	case ComputationEnvironment::CPU:
		memccpy(dst, src, numberOfElements, sizeof(int));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t memcpyVector(FloatPtr dst, const FloatPtr src, int numberOfElements, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMemcpy(dst, src, numberOfElements * sizeof(float), kind);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		break;
	case ComputationEnvironment::CPU:
		memccpy(dst, src, numberOfElements, sizeof(float));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t memcpyVector(DoublePtr dst, const DoublePtr src, int numberOfElements, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMemcpy(dst, src, numberOfElements * sizeof(double), kind);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		break;
	case ComputationEnvironment::CPU:
		memccpy(dst, src, numberOfElements, sizeof(double));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t deviceFree(void *devPtr)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaFree(devPtr);
		break;
	case ComputationEnvironment::CPU:
		free(devPtr);
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}
