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


	int mod = 30000;
	int base = 30000;

	/* initialize random seed: */
	srand(time(NULL));

	string name = argv[1];
	int N = atoi(argv[2]);

	/* init files*/
	string transFileName = name + ".trans2";
	string emissionFileName = name + ".emit2";

	fstream transFile;
	transFile.open(transFileName.c_str(), fstream::out | fstream::in | fstream::trunc);
	fstream emitFile;
	emitFile.open(emissionFileName.c_str(), fstream::out | fstream::in | fstream::trunc);

	vector<int> states;
	states.resize(N, 0);

	/* generate states w1 .... wN*/
	for (int i = 0; i < N; i++)
	{
		states[i] = i + 1;
	}

	vector<int> values;
	values.resize(N, 0);

	/* write values into the files*/
	for (int i = 0; i < N; i++)
	{
		string source = "w" + std::to_string(i + 1);

		int mod = 100;
		double sum = 0;
		double acc = 0;

		for (int j = 0; j < N; j++)
		{
			int val = rand() % (mod - 1) + 1;
			values[j] = val;
			acc += (double)val;
		}

		acc++; // add 1 to avoid numerical instability 

		for (int j = 0; j < N; j++)
		{

			double val = (double)values[j];
			double prob = val / acc;
			sum += prob;

			string target = "w" + std::to_string(j + 1);

			string s_prob = std::to_string(prob);

			transFile << source << " " << target << " " << s_prob << "\n";
		}
		
		assert(sum <= 1);


	}

	transFile.close();
	emitFile.close();
}