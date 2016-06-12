#pragma once

// =================================================================================================
// source: http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
// =================================================================================================

#include "cuda_runtime.h"


//1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D();

//1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D();

//1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D();

//2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D();

//2D grid of 2D blocks
__device__ int getGlobalIdx_2D_2D();

//2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D();

//3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D();

//3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D();

//3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D();