#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>

__host__ void TrellisInitialization2D(double *host_Alpha_trelis_2D, const double *host_Pi_startProbs_1D, const double *host_B_obsEmissionProbs_2D, const unsigned int *host_O_obsSequence_1D, int T_noOfObservations, int N_noOfStates, int V_noOfObsSymbols);


/** Rescales row in trellis identified by time index idx_t to sum of one.
  * @param host_Alpha_trelis_2D the trellis which row should be rescaled
  * @param T_noOfObservations number of rows i.e. time entries in the trellis
  * @param N_noOfStates number of columns i.e. different states in the trellis
  * @param idx_t time index that identifies the affected row in the trellis
  */
__host__ void TrellisScaling2D(double *host_Alpha_trelis_2D, unsigned int T_noOfObservations, unsigned int N_noOfStates, unsigned int idx_t);