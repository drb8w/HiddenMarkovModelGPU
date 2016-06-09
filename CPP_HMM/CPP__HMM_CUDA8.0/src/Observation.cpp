#include "Observation.h"

#include "Utilities.h"

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
				if (peek != '\n'){
					obsFile.read(buffer, 1); // consume whitespace
				}
			}

			else{
				obsFile.read(buffer, 2);
				string t(buffer);
				t.resize(2);
				int i = mapObsToInt(t);
				currentSeq->push_back(i);
				//cout << t << " " << i << "\n";
			}

			peek = obsFile.peek();
		}

		int size = currentSeq->size();

		obsFile.read(buffer, 1); // consume newline
		peek = obsFile.peek();

		if (!obsFile.eof()){
			currentSeq = new vector<unsigned int>();
			sequences.push_back(currentSeq);
		}


	}

	obsFile.close();

	cout << "load Observations end\n";
}

int Observation::mapObsToInt(string obs){

	if (obs.compare("HD") == 0){ // HotDry
		return 0;
	}

	if (obs.compare("HW") == 0){ // HotWet
		return 1;
	}

	if (obs.compare("CD") == 0){ // ColdDry
		return 2;
	}

	if (obs.compare("CW") == 0){ //ColdWet
		return 3;
	}

	return -1;

	}

unsigned int* Observation::observationSequencesAsArray()
{
	unsigned int *M = nullptr;
	int M_noOfObsSequences = this->sequences.size();
	int T_noOfObservations = this->sequences.front()->size();

#ifdef ROW_MAJ_ORD_MAT_ROW_FIRST_INDEX 

	int dim1_M = T_noOfObservations;
	int dim2_M = M_noOfObsSequences;
	M = (unsigned int *)calloc(T_noOfObservations * M_noOfObsSequences, sizeof(unsigned int));

	for (int i = 0; i < dim2_M; i++)
	{
		for (int j = 0; j < dim1_M; j++)
		{
			int idx_m_ij = i*dim1_M + j;
			vector<unsigned int> *obsSequence_i = this->sequences.at(i);
			if (obsSequence_i != nullptr && obsSequence_i->size() > j)
			{
				unsigned int m_ij = this->sequences.at(i)->at(j);
				M[idx_m_ij] = m_ij;
			}
		}
	}

#endif

	return M;

}
