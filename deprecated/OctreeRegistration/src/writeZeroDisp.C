
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

  if(argc < 3) {
    std::cout<<"Usage: exe dispFile Ne "<<std::endl;
    PetscFinalize();
    exit(0);
  }

  int Ne = atoi(argv[2]);

  Vec U;
  VecCreate(PETSC_COMM_SELF, &U);
  VecSetSizes(U, (3*(Ne + 1)*(Ne + 1)*(Ne + 1)), PETSC_DECIDE);
  VecSetType(U, VECSEQ);
  VecZeroEntries(U);

  saveVector(U, argv[1]);
  VecDestroy(U);

  PetscFinalize();
}


