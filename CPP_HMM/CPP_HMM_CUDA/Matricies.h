#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>

using namespace std;

class Matricies {
public:

	const int N = 6; // states
	const int V = 4; // observations

	vector<double> transition; // size: NxN
	vector<double> emission;	// size NxV
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

	double* transitionAsVector(){ return  &transition[0]; }
	double* emissionAsVector(){ return  &emission[0]; }

	~Matricies();
};
		
