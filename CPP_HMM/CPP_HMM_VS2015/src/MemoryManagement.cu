#include "MemoryManagement.cuh"

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>

ComputationEnvironment glob_Env = ComputationEnvironment::GPU;

MemoryMovementDuplication glob_Dup = MemoryMovementDuplication::NO;

__host__ cudaError_t allocateDeviceVector(IntHdl pVector, int numberOfElements, bool cleanAlloc)
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
		if (cleanAlloc)
		{
			cudaStatus = cudaMemset(*pVector, 0, numberOfElements);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemset failed!");
				cudaFree(*pVector);
				*pVector = NULL;
			}
		}
		break;
	case ComputationEnvironment::CPU:
		if (cleanAlloc)
			*pVector = (IntPtr)calloc(numberOfElements, sizeof(int));
		else
			*pVector = (IntPtr)malloc(numberOfElements * sizeof(int));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(UIntHdl pVector, int numberOfElements, bool cleanAlloc)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMalloc((void**)pVector, numberOfElements * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(*pVector);
			*pVector = NULL;
		}
		if (cleanAlloc)
		{
			cudaStatus = cudaMemset(*pVector, 0, numberOfElements);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemset failed!");
				cudaFree(*pVector);
				*pVector = NULL;
			}
		}
		break;
	case ComputationEnvironment::CPU:
		if (cleanAlloc)
			*pVector = (UIntPtr)calloc(numberOfElements, sizeof(unsigned int));
		else
			*pVector = (UIntPtr)malloc(numberOfElements * sizeof(unsigned int));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(FloatHdl pVector, int numberOfElements, bool cleanAlloc)
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
		if (cleanAlloc)
		{
			int factor = sizeof(float) / sizeof(int);
			cudaStatus = cudaMemset(*pVector, 0, numberOfElements * factor);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemset failed!");
				cudaFree(*pVector);
				*pVector = NULL;
			}
		}
		break;
	case ComputationEnvironment::CPU:
		if (cleanAlloc)
			*pVector = (float *)calloc(numberOfElements, sizeof(float));
		else
			*pVector = (float *)malloc(numberOfElements * sizeof(float));
		cudaStatus = cudaError_t::cudaSuccess;
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t allocateDeviceVector(DoubleHdl pVector, int numberOfElements, bool cleanAlloc)
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
		if (cleanAlloc)
		{
			int factor = sizeof(double) / sizeof(int);
			cudaStatus = cudaMemset(*pVector, 0, numberOfElements * factor);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemset failed!");
				cudaFree(*pVector);
				*pVector = NULL;
			}
		}
		break;
	case ComputationEnvironment::CPU:
		if (cleanAlloc)
			*pVector = (double *)calloc(numberOfElements, sizeof(double));
		else
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
		switch (glob_Dup)
		{
		case MemoryMovementDuplication::YES:
			memccpy(dst, src, numberOfElements, sizeof(int));
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		case MemoryMovementDuplication::NO:
			dst = src;
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		}
		break;
	}

	return cudaStatus;
}

__host__ cudaError_t memcpyVector(UIntPtr dst, const UIntPtr src, int numberOfElements, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorIllegalInstruction;
	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		cudaStatus = cudaMemcpy(dst, src, numberOfElements * sizeof(unsigned int), kind);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		break;
	case ComputationEnvironment::CPU:
		switch (glob_Dup)
		{
		case MemoryMovementDuplication::YES:
			memccpy(dst, src, numberOfElements, sizeof(unsigned int));
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		case MemoryMovementDuplication::NO:
			dst = src;
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		}
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
		switch (glob_Dup)
		{
		case MemoryMovementDuplication::YES:
			memccpy(dst, src, numberOfElements, sizeof(float));
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		case MemoryMovementDuplication::NO:
			dst = src;
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		}
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
		switch (glob_Dup)
		{
		case MemoryMovementDuplication::YES:
			memccpy(dst, src, numberOfElements, sizeof(double));
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		case MemoryMovementDuplication::NO:
			dst = src;
			cudaStatus = cudaError_t::cudaSuccess;
			break;
		}
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
