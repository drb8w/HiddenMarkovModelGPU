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
	int V = matricies->V;

	matricies->loadMatricies(argv[1]);
	observations->loadObservations(argv[1]);



	vector<vector<unsigned int>*>* sequences = &observations->sequences;
	int numberOfObservations = sequences->size();

	// for each obs. sequence do
	for (unsigned int i = 0; i<numberOfObservations; i++) {

		cout << "starting fw alg for obs sequence...\n";

		vector<unsigned int>* sequence = sequences->at(i);
		int T = sequence->size();
		int N = matricies->N;

		vector<vector<double>*> trelis;
		trelis.resize(T,new vector<double>());
		for (unsigned int i = 0; i < T; i++){
			trelis.at(i)->resize(N, 0);
		}

		int startingObs = sequence->at(0);
		
		//init the trelis
		for (unsigned int i = 0; i < N; i++){
			double initVal = matricies->pi[i] + matricies->emission[i*V + startingObs];
			trelis.at(0)->at(i) = initVal;
		}

		for (unsigned int i = 1; i < T; i++){

			// call kernel for NxV matrix ops (N is the number of states, V is the number of observations)
			// Launch a kernel on the GPU with one thread for each element.
			//fwKernel <<<N, V >>>(transProbMatrix, emissionProbMatrix, t);

		}


	}

	cout << "end\n";

	return 0;
}

__global__ void fwKernel(){

}