#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

#include "hmm.h"

int main(int argc, char* argv[])
{
  Hmm hmm;

  if (argc<2) {
    cerr << "USAGE: genseq NAME N L" << endl
	 << "generates N observation sequences with length L using the HMM with the given NAME. L = 0 for normal usage" << endl;
    exit(1);
  }
  hmm.loadProbs(argv[1]);
  int seqs = atoi(argv[2]);
  int length = atoi(argv[3]);
  if (length == 0){
	  hmm.genSeqs(cout, seqs);
  }
  else{
	  hmm.genSeqsFixedLength(cout, seqs, length);
  }


}
  
