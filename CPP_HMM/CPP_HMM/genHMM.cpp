#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include <assert.h>
using namespace std;

int main(int argc, char* argv[])
{

	if (argc<2) {
		cerr << "USAGE: genHMM NAME N" << endl
			<< "genreats a HMM with name NAME and N states and random non-zero probabilities" << endl;
		exit(1);
	}

	string name = argv[1];
	int N = atoi(argv[2]);

	string transFileName = name + ".trans2";
	string emissionFileName = name + ".emit2";

	fstream transFile(transFileName.c_str(), std::ifstream::in | std::ifstream::out);
	fstream emitFile(emissionFileName.c_str(), std::ifstream::in | std::ifstream::out);

	transFile << "test" << "anotherTest";

	transFile.close();
	emitFile.close();
}