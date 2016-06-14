#include "Benchmark.h"

void initBenchmark(cudaEvent_t* start, cudaEvent_t* stop){

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(start);
	cudaEventCreate(stop);

}

void startBenchmark(cudaEvent_t start){

	// Record the start event
	cudaEventRecord(start, NULL);
}

void stopBenchmark(char* name, cudaEvent_t start, cudaEvent_t stop){

	// Record the stop event
	cudaEventRecord(stop, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

	float msecPerIteration = msecTotal / ITERATIONS;
	printf("Performance for %s = %.3f msec\n",name, msecPerIteration);
}