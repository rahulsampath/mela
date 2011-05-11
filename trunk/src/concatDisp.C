
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
   MPI_Comm commAll = MPI_COMM_WORLD;


   int rank, npesAll;
    MPI_Comm_rank(commAll, &rank);
    MPI_Comm_size(commAll, &npesAll);

        if (!rank){
 
        concatDispFile(argv[1],npesAll,argv[2]);
  
    }
/*
  if(argc < 4) {
    std::cout<<"Usage: exe disp1File disp2File dispOutFile"<<std::endl;
    PetscFinalize();
    exit(0);
  }

  Vec U1, U2;
  loadSeqVector(U1, argv[1]);
  loadSeqVector(U2, argv[2]);

  concatSaveVector(U1, U2, argv[3]);

  VecDestroy(U1);
  VecDestroy(U2);
*/
  PetscFinalize();
}

