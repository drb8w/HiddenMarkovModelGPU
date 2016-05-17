#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matricies.h"
#include "Observation.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{

	cout << "start...\n";
	Matricies* matricies = new Matricies();
	Observation* observations = new Observation();

	matricies->loadMatricies(argv[1]);
	observations->loadObservations(argv[1]);

	vector<vector<unsigned int>*>* sequences = &observations->sequences;
	int numberOfObservations = sequences->size();

	// for each obs. sequence do
	for (unsigned int i = 0; i<numberOfObservations; i++) {

		vector<unsigned int>* sequence = sequences->at(i);
		int observations = sequence->size();

		for (unsigned int i = 0; i < observations; i++){
			// call kernel for NxN matrix ops (N is the number of states)
			// Launch a kernel on the GPU with one thread for each element.
			//fwKernel <<<N, N >>>(transProbMatrix, emissionProbMatrix, t);

		}


	}

	cout << "end\n";

	return 0;
}