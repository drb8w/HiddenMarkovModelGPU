#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.cuh"
#include "Utilities.h"
#include "VectorMath.cuh"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

// ------------------------------------------------------------------------------------------------------
// global states
// ------------------------------------------------------------------------------------------------------
extern ComputationEnvironment glob_Env;

int main(int argc, char* argv[])
{

	// dim 3x4 ( r x c )
	double A[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

	// dim 4x3 ( r x c )
	double B[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

	// result matrix ; dim 3x3
	double *C = nullptr;
	glob_Env = ComputationEnvironment::CPU;
	allocateDeviceVector(&C, 9, true);
	glob_Env = ComputationEnvironment::GPU;

	double *dev_A = nullptr;
	double *dev_B = nullptr;
	double *dev_C = nullptr;

	allocateDeviceVector(&dev_A, 12,true);
	allocateDeviceVector(&dev_B, 12,true);
	allocateDeviceVector(&dev_C, 9,true);

	double* A_start = &A[0];
	double* B_start = &B[0];

	memcpyVector(dev_A, A_start, 12, cudaMemcpyHostToDevice);
	memcpyVector(dev_B, B_start, 12, cudaMemcpyHostToDevice);

	cublasMultiplyDouble(3, 3, 4, dev_A, dev_B, dev_C);

	memcpyVector(C, dev_C, 9, cudaMemcpyDeviceToHost);
	memcpyVector(A, dev_A, 12, cudaMemcpyDeviceToHost);
	memcpyVector(B, dev_B, 12, cudaMemcpyDeviceToHost);

	deviceFree(dev_A);
	deviceFree(dev_B);
	deviceFree(dev_C);

	/*
	 * 70  80  90
	 * 158 184 210
	 * 246 288 330
	 */

	for (int i = 0; i < 9; i++)
	{
		cout << C[i] << " ";
	}



}