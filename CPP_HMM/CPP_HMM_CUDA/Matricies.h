#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include <assert.h>

using namespace std;

class Matricies {
public:

	const int N = 16; // number of states
	const int V = 4; // number of observation symbols

	vector<double> transition; // size: NxN
	vector<double> emission;	// size NxV
	// is part of the transition probability ???
	vector<double> pi; // size N

	Matricies(){
		transition.resize(N*N,0.0);
		emission.resize(N*V, 0.0);
		initPiMatrix();
	}

	void loadMatricies(string fileName);
	void initPiMatrix(){
		pi.resize(N, 0.0);

		int mod = 30000;
		double acc = 0;
		double sum = 0;

		vector<int> values;
		values.resize(N, 0);

		/* initialize random seed: */
		srand(time(NULL));

		for (int i = 0; i < N; i++)
		{
			int val = rand() % (mod - 1) + 1;
			values[i] = val;
			acc += (double)val;
		}

		acc++; // add 1 to avoid numerical instability 

		// basic init
		for (int i = 0; i < N; i++)
		{
			double val = (double)values[i];
			double prob = val / acc;
			sum += prob;
			pi[i] = prob;


		}

		assert(sum <= 1);

		//pi[0] = 0.45; // sunny
		//pi[1] = 0.1; // cloudy
		//pi[2] = 0.3; // rainy
		//pi[3] = 0.05; // snow 
		//pi[4] = 0.05; // stor 
		//pi[5] = 0.05; // haze
	}

	double* piAsArray(){ return &pi[0]; }
	double* transitionAsArray(){ return  &transition[0]; }
	double* emissionAsArray(){ return  &emission[0]; }

	~Matricies();
};
		
