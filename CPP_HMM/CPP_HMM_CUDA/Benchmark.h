#pragma once

#include "cuda_runtime.h"
#include "MemoryManagement.cuh"

#include <stdio.h>
#include <string>
#include <ctime>

using namespace std;

const int ITERATIONS = 1000;

void initBenchmark(cudaEvent_t* start, cudaEvent_t* stop);

void startBenchmark(cudaEvent_t start, clock_t* start_time);

void stopBenchmark(char* name, cudaEvent_t start, cudaEvent_t stop, clock_t* start_time, clock_t* end_time, ComputationEnvironment env);


