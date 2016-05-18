#include "Matricies.h"

void Matricies::loadMatricies(string filename){

	cout << "load matricies start\n";

	string transFileName = filename + ".trans2";
	string emissionFileName = filename + ".emit2";

	ifstream transFile(transFileName.c_str());
	ifstream emitFile(emissionFileName.c_str());

	string name1, name2;
	double c;

	int index = 0;

	while (transFile >> name1 >> name2 >> c) {
		transition[index] = c;	
	}

	index = 0;

	while (emitFile >> name1 >> name2 >> c) {
		emission[index] = c;
	}

	transFile.close();
	emitFile.close();

	cout << "load matricies end\n";

}