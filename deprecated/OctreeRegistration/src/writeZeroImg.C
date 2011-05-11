
#include "mpi.h"
#include "registration.h"
#include "externVars.h"
#include "dendro.h"
#include <iostream>

int elasMultEvent;
int hessMultEvent;
int hessFinestMultEvent;
int createHessContextEvent;
int updateHessContextEvent;
int computeSigEvent;
int computeTauEvent;
int computeGradTauEvent;
int computeNodalTauEvent;
int computeNodalGradTauEvent;
int evalObjEvent;
int evalGradEvent;
int createPatchesEvent;
int expandPatchesEvent;
int meshPatchesEvent;
int copyValsToPatchesEvent;
int optEvent;

int main(int argc, char** argv) {

  PetscInitialize(&argc, &argv, 0, 0);

  int Ne = 256;
  
  if(argc > 1) {
    Ne = atoi(argv[1]);
  }

  std::vector<double> img(Ne*Ne*Ne);

  for(int i = 0; i < img.size(); i++) {
    img[i] = 0.0;
  }
  
  writeImage("zeroImg", Ne, Ne, Ne, img);

  PetscFinalize();
}


