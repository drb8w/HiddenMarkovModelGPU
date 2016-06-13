#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

#include "hmm.h"

int main(int argc, char* argv[])
{
  Hmm hmm;
  //if (argc<3) {
  //  cerr << "USAGE: trainhmm INIT-HMM RESULT-HMM DATA [MAX-ITERATIONS]" << endl;
  //  exit(1);
  //}

  hmm.loadProbs("test");
  const char* output = "test";
  ifstream istrm("test.train");
  int maxIterations = 10;
  if (argc>4)
    maxIterations = atoi(argv[4]);

  vector<vector<unsigned long>*> trainingSequences;
  hmm.readSeqs(istrm, trainingSequences);
  hmm.baumWelch(trainingSequences, maxIterations);
  hmm.saveProbs(output);
}

