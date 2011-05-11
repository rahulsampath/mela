
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

  if(argc < 6) {
    std::cout<<"Usage: exe dispFile dispScale inpImgFile outImgFile computeDetJac"<<std::endl;
    PetscFinalize();
    exit(0);
  }

  bool computeDetJac = (bool)(atoi(argv[5]));

  Vec U;
  loadSeqVector(U, argv[1]) ;
  double dispScale = atof(argv[2]);
  VecScale(U, dispScale);

  struct dsr hdr;
  std::vector<double> img;
  readImage(argv[3], &hdr, img);

  int Ne = hdr.dime.dim[1];
  assert(hdr.dime.dim[2] == Ne);
  assert(hdr.dime.dim[3] == Ne);

  assert(img.size() == (Ne*Ne*Ne));

  Vec imgElemental;
  VecCreate(PETSC_COMM_SELF, &imgElemental);
  VecSetSizes(imgElemental, (Ne*Ne*Ne), PETSC_DECIDE);
  VecSetType(imgElemental, VECSEQ);

  std::cout<<"Passed Stage 1"<<std::endl;

  PetscScalar* arr;
  VecGetArray(imgElemental, &arr);

  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        int idx = (((k*Ne) + j)*Ne) + i;
        arr[idx] = img[idx];
      }
    }
  }

  VecRestoreArray(imgElemental, &arr);
  img.clear();

  DA da1dof;
  DA da3dof;
  Vec imgNodal;
  Vec gradImg;

  DACreate3d(PETSC_COMM_SELF, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da1dof);

  DACreate3d(PETSC_COMM_SELF, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da3dof);

  std::cout<<"Passed Stage 2"<<std::endl;

  DACreateGlobalVector(da1dof, &imgNodal);
  DACreateGlobalVector(da3dof, &gradImg);

  PetscScalar* elemArr;
  PetscScalar* nodalArr;

  VecGetArray(imgElemental, &elemArr);
  VecGetArray(imgNodal, &nodalArr);

  for(int k = 0 ; k < Ne; k++) {
    for(int j = 0 ; j < Ne; j++) {
      for(int i = 0 ; i < Ne; i++) {
        nodalArr[(((k*(Ne + 1)) + j)*(Ne + 1)) + i] = elemArr[(((k*Ne) + j)*Ne) + i];
      }
    }
  }

  for(int k = 0 ; k < Ne; k++) {
    for(int j = 0 ; j < Ne; j++) {
      nodalArr[(((k*(Ne + 1)) + j)*(Ne + 1)) + Ne] = nodalArr[(((k*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1];
    }
  }

  for(int k = 0 ; k < Ne; k++) {
    for(int i = 0 ; i < Ne; i++) {
      nodalArr[(((k*(Ne + 1)) + Ne)*(Ne + 1)) + i] = nodalArr[(((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i];
    }
  }

  for(int j = 0 ; j < Ne; j++) {
    for(int i = 0 ; i < Ne; i++) {
      nodalArr[(((Ne*(Ne + 1)) + j)*(Ne + 1)) + i] = nodalArr[((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + i];
    }
  }

  for(int i = 0 ; i < Ne; i++) {
    nodalArr[(((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + i] = nodalArr[((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i];
  }

  for(int j = 0 ; j < Ne; j++) {
    nodalArr[(((Ne*(Ne + 1)) + j)*(Ne + 1)) + Ne] = nodalArr[((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1];
  }

  for(int k = 0 ; k < Ne; k++) {
    nodalArr[(((k*(Ne + 1)) + Ne)*(Ne + 1)) + Ne] = nodalArr[(((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1];
  }

  nodalArr[(((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + Ne] = nodalArr[((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1];

  VecRestoreArray(imgElemental, &elemArr);
  VecRestoreArray(imgNodal, &nodalArr);

  VecDestroy(imgElemental);

  if(computeDetJac) {
    double maxDetJac, minDetJac;
    detJacMaxAndMin(da3dof,  U, &maxDetJac, &minDetJac);
    std::cout<<"DetJac: max: "<<maxDetJac<<" min: "<<minDetJac<<std::endl;
  }

  std::cout<<"Passed Stage 3"<<std::endl;

  computeFDgradient(da1dof, da3dof, imgNodal, gradImg);

  std::cout<<"Passed Stage 4"<<std::endl;

  std::vector<std::vector<double> > tau(1);
  std::vector<std::vector<double> > gradTau(1);
  std::vector<std::vector<double> > imgFinalNodal;

  PetscScalar* imgArr;
  PetscScalar* gradArr;
  VecGetArray(imgNodal, &imgArr);
  VecGetArray(gradImg, &gradArr);

  tau[0].resize((Ne + 1)*(Ne + 1)*(Ne + 1));
  gradTau[0].resize(3*(Ne + 1)*(Ne + 1)*(Ne + 1));

  for(int k = 0; k < (Ne + 1); k++) {
    for(int j = 0; j < (Ne + 1); j++) {
      for(int i = 0; i < (Ne + 1); i++) {
        int ptCnt = (((k*(Ne + 1)) + j)*(Ne + 1)) + i;
        tau[0][ptCnt] = imgArr[ptCnt];
        for(int d = 0; d < 3; d++) {
          gradTau[0][(3*ptCnt) + d] = gradArr[(3*ptCnt) + d];
        }
      }
    }
  }

  VecRestoreArray(imgNodal, &imgArr);
  VecRestoreArray(gradImg, &gradArr);

  VecDestroy(imgNodal);
  VecDestroy(gradImg);

  std::cout<<"Passed Stage 5"<<std::endl;

  computeRegularNodalTauAtU(Ne, tau, gradTau, U, imgFinalNodal);

  std::cout<<"Passed Stage 6"<<std::endl;

  tau.clear();
  gradTau.clear();

  VecDestroy(U);

  std::vector<double> imgFinal(Ne*Ne*Ne);

  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        imgFinal[(((k*Ne) + j)*Ne) + i] = imgFinalNodal[0][(((k*(Ne + 1)) + j)*(Ne + 1)) + i];
      }
    }
  }

  imgFinalNodal.clear();

  //Remove out of range values Image
  //Convert to integer values
  for(int i = 0; i < imgFinal.size(); i++) {
    if(imgFinal[i] < 0.0) {
      imgFinal[i] = 0.0;
    }
    if(imgFinal[i] > 255.0) {
      imgFinal[i] = 255.0;
    }
    imgFinal[i] = floor(imgFinal[i]);
  }

  std::cout<<"Passed Stage 7"<<std::endl;

  writeImage(argv[4], Ne, Ne, Ne, imgFinal);

  std::cout<<"Passed Stage 8"<<std::endl;

  PetscFinalize();
}



