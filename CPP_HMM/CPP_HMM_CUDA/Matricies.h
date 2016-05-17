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

	Matricies(){
		transition.resize(N*N,0.0);
		emission.resize(N*V, 0.0);
	}

	void loadMatricies(string fileName);

	~Matricies();
};
		
