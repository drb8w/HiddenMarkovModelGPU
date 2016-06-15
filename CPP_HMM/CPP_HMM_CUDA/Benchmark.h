#pragma once

#include "cuda_runtime.h"

#include <stdio.h>
#include <string>

using namespace std;

const int ITERATIONS = 1;

void initBenchmark(cudaEvent_t* start, cudaEvent_t* stop);

void startBenchmark(cudaEvent_t start);

void stopBenchmark(char* name, cudaEvent_t start,cudaEvent_t stop);

