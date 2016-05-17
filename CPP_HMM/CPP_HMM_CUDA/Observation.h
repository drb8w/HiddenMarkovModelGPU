#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>

using namespace std;

class Observation{

public:
	vector<vector<unsigned int>*> sequences;
	Observation(){};
	~Observation(){
		for (int i = 0; i < sequences.size(); i++){
			delete sequences[i];
		}
	};
	
	void loadObservations(string filename);
	int mapObsToInt(string obs);
};