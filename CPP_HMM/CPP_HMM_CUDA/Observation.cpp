#include "Observation.h"

void Observation::loadObservations(string filename){

	cout << "load Observations start\n";

	string obsFilename = filename + ".input";
	ifstream obsFile(obsFilename.c_str());

	char * buffer = new char[2];
	vector <unsigned int>* currentSeq = new vector<unsigned int>();
	sequences.push_back(currentSeq);

	char peek = obsFile.peek();

	while (!obsFile.eof()){

		while (peek != '\n' && !obsFile.eof()){

			if (peek == ' ' || peek == '\t' || peek == '\n'){
				obsFile.read(buffer, 1); // consume whitespace
			}

			else{
				obsFile.read(buffer, 2);
				string t(buffer);
				currentSeq->push_back(mapObsToInt(t));
			}

			peek = obsFile.peek();
		}

		obsFile.read(buffer, 1); // consume newline

		vector <unsigned int>* currentSeq = new vector<unsigned int>();
		sequences.push_back(currentSeq);
	}

	obsFile.close();

	cout << "load Observations end\n";
}

int Observation::mapObsToInt(string obs){

	if (obs.compare("HD")){ // HotDry
		return 0;
	}

	if (obs.compare("HW")){ // HotWet
		return 1;
	}

	if (obs.compare("CD")){ // ColdDry
		return 2;
	}

	if (obs.compare("CW")){ //ColdWet
		return 3;
	}

	return -1;

	}
