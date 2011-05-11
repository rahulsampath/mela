
#include "mpi.h"
#include "registration.h"
#include "externVars.h"
#include "dendro.h"
#include <iostream>

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

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

double sphereImgVal(double x, double y, double z, int isoFac) {
  double hFac = pow(0.5, static_cast<double>(isoFac));

  double xOff = hFac*(floor(x/hFac));
  double yOff = hFac*(floor(y/hFac));
  double zOff = hFac*(floor(z/hFac));

  double xNew = (x - xOff)/hFac;
  double yNew = (y - yOff)/hFac;
  double zNew = (z - zOff)/hFac;

  double radSqr = SQR(0.25);
  double distSqr = (SQR(xNew - 0.5)) + (SQR(yNew - 0.5)) + (SQR(zNew - 0.5)); 
  double imgVal;

  if(distSqr > radSqr) {
    //out
    imgVal = 0.0;
  } else  {
    //in
    imgVal = 255.0;
  }

  return imgVal;
}

double cImgVal(double x, double y, double z, int isoFac) {
  double hFac = pow(0.5, static_cast<double>(isoFac));

  double xOff = hFac*(floor(x/hFac));
  double yOff = hFac*(floor(y/hFac));
  double zOff = hFac*(floor(z/hFac));

  double xNew = (x - xOff)/hFac;
  double yNew = (y - yOff)/hFac;
  double zNew = (z - zOff)/hFac;

  bool in = false;
  if( (SQR(0.25 - sqrt((SQR(xNew - 0.5)) + (SQR(yNew - 0.5)))))
      + (SQR(zNew - 0.5)) <= (SQR(0.1)) ) {
    if(xNew < 0.6) {
      in = true;
    }
  }

  double imgVal;

  if(in) {
    imgVal = 255.0;
  } else {
    imgVal = 0.0;
  }

  return imgVal;
}

void createSphereIsoFacImage(int Ne, int isoFac, Vec v) {
  //This is sequential
  VecZeroEntries(v);

  PetscScalar* arr;
  VecGetArray(v, &arr);

  double h = 1.0/static_cast<double>(Ne);

  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        double x = static_cast<double>(i)*h;
        double y = static_cast<double>(j)*h;
        double z = static_cast<double>(k)*h;
        arr[(((k*Ne) + j)*Ne) + i] = sphereImgVal(x, y, z, isoFac);
      }//end for i
    }//end for j
  }//end for k

  VecRestoreArray(v, &arr);
}

void createCshapeIsoFacImage(int Ne, int isoFac, Vec v) {
  //This is sequential
  VecZeroEntries(v);

  PetscScalar* arr;
  VecGetArray(v, &arr);

  double h = 1.0/static_cast<double>(Ne);

  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        double x = static_cast<double>(i)*h;
        double y = static_cast<double>(j)*h;
        double z = static_cast<double>(k)*h;
        arr[(((k*Ne) + j)*Ne) + i] = cImgVal(x, y, z, isoFac);
      }//end for i
    }//end for j
  }//end for k

  VecRestoreArray(v, &arr);
}

int main(int argc, char** argv) {

  PetscInitialize(&argc, &argv, 0, 0);

  if(argc < 4) {
    std::cout<<"Usage: exe Ne isoFac imgType outFile."<<std::endl;
    PetscFinalize();
    exit(0);
  }

  int Ne = atoi(argv[1]);
  int isoFac = atoi(argv[2]); 
  int imgType = atoi(argv[3]);

  Vec imgVec;
  VecCreate(PETSC_COMM_SELF, &imgVec);
  VecSetSizes(imgVec, (Ne*Ne*Ne), PETSC_DECIDE);
  VecSetType(imgVec, VECSEQ);

  if(imgType == 0) {
    createSphereIsoFacImage(Ne, isoFac, imgVec) ;
  } else {
    createCshapeIsoFacImage(Ne, isoFac, imgVec) ;
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

