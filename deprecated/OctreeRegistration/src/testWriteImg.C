
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

  if(argc < 5) {
    std::cout<<"Usage: exe Ne width imgType outFile."<<
      " imgType = 0 for chkBoard."<<
      " imgType = 1 for sin^2."<<
      " imgType = 2 for sphere."<<
      " imgType = 3 for C-shape."<<
      " imgType = 4 for Fine Anal Moving."<<
      " imgType = 5 for Fine Anal Fixed."<<std::endl;
    PetscFinalize();
    exit(0);
  }

  int Ne = atoi(argv[1]);
  int width = atoi(argv[2]); 
  int imgType = atoi(argv[3]);

  Vec imgVec;
  VecCreate(PETSC_COMM_SELF, &imgVec);
  VecSetSizes(imgVec, (Ne*Ne*Ne), PETSC_DECIDE);
  VecSetType(imgVec, VECSEQ);

  if(imgType == 0) {
    createCheckerBoardImage(Ne, width, imgVec);
  } else if(imgType == 1) {
    createAnalyticImage(Ne, width, imgVec);
  } else if(imgType == 2) {
    createSphereImage(Ne, width, imgVec);
  } else if(imgType == 3) {
    createCshapeImage(Ne, width, imgVec);
  } else if(imgType == 4) {
    createFineAnalyticImage(Ne, width, imgVec);
  } else if(imgType == 5) {
    createFineAnalyticFixedImage(Ne, width, imgVec);
  } else {
    assert(false);
  }

  std::vector<double> img(Ne*Ne*Ne);

  PetscScalar* arr;
  VecGetArray(imgVec, &arr);

  int cnt = 0;
  for(int k = 0 ; k < Ne; k++) {
    for(int j = 0 ; j < Ne; j++) {
      for(int i = 0 ; i < Ne; i++) {
        img[cnt] = arr[cnt];
        cnt++;
      }
    }
  }

  VecRestoreArray(imgVec, &arr);

  VecDestroy(imgVec);

  writeImage(argv[4], Ne, Ne, Ne, img);

  PetscFinalize();
}

