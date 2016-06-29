#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include <assert.h>
#include<windows.h>
using namespace std;

int main(int argc, char* argv[])
{

	if (argc<3) {
		cerr << "USAGE: genHMM NAME N B" << endl
			<< "genreats a HMM with name NAME and N states and random non-zero probabilities and B observations" << endl;
		exit(1);
	}

	/* initialize random seed: */
	//srand(time(NULL));
	// to have reproducible results
	srand(0);

	string name = argv[1];
	int N = atoi(argv[2]);
	int B = atoi(argv[3]);

	string path = "../CPP_HMM_CUDA/" + name;

	CreateDirectory(path.c_str(), NULL);

	/* init files*/
	string stateFileName = path + "/" + name + ".state2";
	string transFileName = path + "/" + name + ".trans2";
	string emissionFileName = path + "/" + name + ".emit2";

	fstream stateFile;
	stateFile.open(stateFileName.c_str(), fstream::out | fstream::in | fstream::trunc);
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

			if (j == (N - 1) && i == (N - 1)){
				transFile << source << " " << target << " " << s_prob;
			}
			else{
				transFile << source << " " << target << " " << s_prob << "\n";
			}

		}
		
		assert(sum <= 1);


	}

	values.resize(B, 0);

	for (int i = 0; i < N; i++)
	{
		string state = "w" + std::to_string(i + 1);

		int mod = 100;
		double sum = 0;
		double acc = 0;

		for (int j = 0; j < B; j++)
		{
			int val = rand() % (mod - 1) + 1;
			values[j] = val;
			acc += (double)val;
		}

		acc++; // add 1 to avoid numerical instability 

		for (int  j = 0; j < B; j++)
		{
			double val = (double)values[j];
			double prob = val / acc;
			sum += prob;

			string s_prob = std::to_string(prob);

			string obs = "o" + std::to_string(j + 1);

			if (j == (B - 1) && i == (N - 1)){
				emitFile << state << " " << obs << " " << s_prob;
			}
			else{
				emitFile << state << " " << obs << " " << s_prob << "\n";
			}


		}

		assert(sum <= 1);
	}

	transFile.close();
	emitFile.close();
}