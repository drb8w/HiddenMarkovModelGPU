#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>

using namespace std;

class Matricies {
public:

	const int N = 6; // number of states
	const int V = 4; // number of observation symbols

	vector<double> transition; // size: NxN
	vector<double> emission;	// size NxV
	// is part of the transition probability ???
	vector<double> pi; // size N

	Matricies(){
		transition.resize(N*N,0.0);
		emission.resize(N*V, 0.0);
		pi.resize(N, 0.0);

		pi[0] = 0.45; // sunny
		pi[1] = 0.1; // cloudy
		pi[2] = 0.3; // rainy
		pi[3] = 0.05; // snow 
		pi[4] = 0.05; // stor 
		pi[5] = 0.05; // haze
	}

	void loadMatricies(string fileName);

	double* piAsArray(){ return &pi[0]; }
	double* transitionAsArray(){ return  &transition[0]; }
	double* emissionAsArray(){ return  &emission[0]; }

	~Matricies();
};
		
