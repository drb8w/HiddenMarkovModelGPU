#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include <assert.h>
#include <math.h>

using namespace std;

class Matricies {
public:

	int N = 0; // number of states
	int V = 0; // number of observation symbols

	vector<double> transition; // size: NxN
	vector<double> emission;	// size NxV
	// is part of the transition probability ???
	vector<double> pi; // size N

	Matricies(string fileName){

		string transFileName = fileName + ".trans2";
		string emissionFileName = fileName + ".emit2";

		ifstream transFile(transFileName.c_str());
		ifstream emitFile(emissionFileName.c_str());

		int count = 1;
		char x;
		while (transFile.get(x)){
			if (x == '\n')
				count++;
		}

		N = sqrt(count);

		count = 1;
		while (emitFile.get(x)){
			if (x == '\n')
				count++;
		}

		V = count/N;

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
		//srand(time(NULL));
		// fix seed for reproducibility
		srand(0);

		for (int i = 0; i < N; i++)
		{
			int val = rand() % (mod - 1) + 1;
			values[i] = val;
			acc += (double)val;
		}

		acc++; // add 1 to avoid numerical instability 

		for (int i = 0; i < N; i++)
		{
			double val = (double)values[i];
			double prob = val / acc;
			sum += prob;
			pi[i] = prob;

		}

		assert(sum <= 1);
	}

	double* piAsArray(){ return &pi[0]; }
	double* transitionAsArray(){ return  &transition[0]; }
	double* emissionAsArray(){ return  &emission[0]; }

	~Matricies();
};
		
