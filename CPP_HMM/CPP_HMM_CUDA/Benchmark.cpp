#include "Benchmark.h"

extern ComputationEnvironment glob_Env;

void initBenchmark(cudaEvent_t* start, cudaEvent_t* stop){

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		// Allocate CUDA events that we'll use for timing
		cudaEventCreate(start);
		cudaEventCreate(stop);
		break;
	case ComputationEnvironment::CPU:
		break;
	}


}

void startBenchmark(cudaEvent_t start, clock_t* start_time){

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		// Record the start event
		cudaEventRecord(start, NULL);
		break;
	case ComputationEnvironment::CPU:
		*start_time = clock();
		break;
	}
}

void stopBenchmark(char* name, cudaEvent_t start, cudaEvent_t stop, clock_t* start_time, clock_t* end_time){

	float msecTotal = 0.0f;
	float msecPerIteration = 0.0f;
	double diffSec = 0;
	float msecPerIterationF = 0.0f;
	double msecPerIterationD = 0;

	switch (glob_Env)
	{
	case ComputationEnvironment::GPU:
		// Record the stop event
		cudaEventRecord(stop, NULL);

		// Wait for the stop event to complete
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&msecTotal, start, stop);

		msecPerIterationF = msecTotal / ITERATIONS;
		printf("Performance for %s = %.3f msec\n", name, msecPerIterationF);
		break;
	case ComputationEnvironment::CPU:
		*end_time = clock();
		diffSec = double(*end_time- *start_time)/ CLOCKS_PER_SEC;
		msecPerIterationD = (diffSec / ITERATIONS) * 1000;
		printf("Performance for %s = %.3f msec\n", name, msecPerIterationD);

		break;
	}

	
}