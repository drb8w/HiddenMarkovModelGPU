#pragma once

#include "cuda_runtime.h"

// ------------------------------------------------------------------------------------------------------
// global states
// ------------------------------------------------------------------------------------------------------

enum ComputationEnvironment{ CPU, GPU };

//ComputationEnvironment glob_Env = ComputationEnvironment::GPU;

// ------------------------------------------------------------------------------------------------------
// typedefs
// ------------------------------------------------------------------------------------------------------
typedef int *IntPtr;
typedef IntPtr *IntHdl;

typedef float *FloatPtr;
typedef FloatPtr *FloatHdl;

typedef double *DoublePtr;
typedef DoublePtr *DoubleHdl;

// ------------------------------------------------------------------------------------------------------
// declarations
// ------------------------------------------------------------------------------------------------------

__host__ cudaError_t allocateDeviceVector(IntHdl pVector, int numberOfElements, bool cleanAlloc = false);
__host__ cudaError_t allocateDeviceVector(FloatHdl pVector, int numberOfElements, bool cleanAlloc = false);
__host__ cudaError_t allocateDeviceVector(DoubleHdl pVector, int numberOfElements, bool cleanAlloc = false);

__host__ cudaError_t memcpyVector(IntPtr dst, const IntPtr src, int numberOfElements, enum cudaMemcpyKind kind);
__host__ cudaError_t memcpyVector(FloatPtr dst, const FloatPtr src, int numberOfElements, enum cudaMemcpyKind kind);
__host__ cudaError_t memcpyVector(DoublePtr dst, const DoublePtr src, int numberOfElements, enum cudaMemcpyKind kind);

__host__ cudaError_t deviceFree(void *devPtr);