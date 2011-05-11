
#include "mpi.h"
#include "regInterpCubic.h"
#include "registration.h"
#include <cmath>
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

  if(argc < 4) {
    std::cout<<"Usage: exe disp1File disp2File dispOutFile"<<std::endl;
    PetscFinalize();
    exit(0);
  }

  Vec U1, U2, U3;
  loadSeqVector(U1, argv[1]);
  loadSeqVector(U2, argv[2]);

  VecDuplicate(U1, &U3);

  VecWAXPY(U3, 1.0, U1, U2);

  saveVector(U3, argv[3]);

  VecDestroy(U1);
  VecDestroy(U2);
  VecDestroy(U3);

  PetscFinalize();
}

