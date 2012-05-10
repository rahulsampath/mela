
#include "mpi.h"
#include "sys/sys.h"
#include "omg/omg.h"
#include "registration.h"
#include "regInterpCubic.h"
#include "externVars.h"
#include "dendro.h"

#define __PI__ 3.14159265

#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

int elasMultEvent;
int hessMultEvent;
int hessFinestMultEvent;
int createHessContextEvent;
int updateHessContextEvent;
int evalObjEvent;
int evalGradEvent;
int createPatchesEvent;
int optEvent;
int computeSigEvent;
int computeTauEvent;
int computeGradTauEvent;
int computeNodalTauEvent;
int computeNodalGradTauEvent;
int tauElemAtUEvent;







int main(int argc, char **argv){
  int numImg=6;
  PetscInitialize(&argc,&argv,"options" ,0);

  PetscOptionsGetInt(0,"-numImages", &numImg,0);
 
  writeVecImg(argv[1],argv[2],numImg);
  PetscFinalize();

}
