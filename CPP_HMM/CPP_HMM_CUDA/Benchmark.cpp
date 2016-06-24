#include "Benchmark.h"


void initBenchmark(cudaEvent_t* start, cudaEvent_t* stop){

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(start);
	cudaEventCreate(stop);


}

void startBenchmark(cudaEvent_t start, clock_t* start_time){

	cudaEventRecord(start, NULL);
	*start_time = clock();

}

void stopBenchmark(char* name, cudaEvent_t start, cudaEvent_t stop, clock_t* start_time, clock_t* end_time){

	float msecTotal = 0.0f;
	float msecPerIteration = 0.0f;
	double diffSec = 0;
	float msecPerIterationF = 0.0f;
	double msecPerIterationD = 0;


	// Record the stop event
	cudaEventRecord(stop, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&msecTotal, start, stop);

	msecPerIterationF = msecTotal / ITERATIONS;
	//printf("Performance for %s = %.3f msec\n", name, msecPerIterationF);

	*end_time = clock();
	diffSec = double(*end_time- *start_time)/ CLOCKS_PER_SEC;
	msecPerIterationD = (diffSec / ITERATIONS) * 1000;
	printf("Performance for %s = %.3f msec\n", name, msecPerIterationD + msecPerIterationF);


	
}