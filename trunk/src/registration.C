
/**
  @file registration.C
  @brief Components of Elastic Registration
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  */

#include "mpi.h"
#include "oda/oda.h"
#include "omg/omg.h"
#include "petscda.h"
#include <vector>
#include "par/parUtils.h"
#include "seq/seqUtils.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include "registration.h"
#include "regInterpCubic.h"
#include "dbh.h"

#define DiffX(cArr, psi, eta, gamma) ((cArr[1]) + ((cArr[4])*(eta)) + \
    ((cArr[6])*(gamma)) + ((cArr[7])*(eta)*(gamma)))

#define DiffY(cArr, psi, eta, gamma) ((cArr[2]) + ((cArr[4])*(psi)) + \
    ((cArr[5])*(gamma)) + ((cArr[7])*(psi)*(gamma)))

#define DiffZ(cArr, psi, eta, gamma) ((cArr[3]) + ((cArr[5])*(eta)) + \
    ((cArr[6])*(psi)) + ((cArr[7])*(eta)*(psi)))

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

extern int hessMultEvent;
extern int hessFinestMultEvent;
extern int createPatchesEvent;
extern int createHessContextEvent;
extern int updateHessContextEvent;
extern int elasMultEvent;
extern int evalObjEvent;
extern int evalGradEvent;
extern int optEvent;


namespace ot {
  extern double**** ShapeFnCoeffs;
}

void processImgNatural(DA da1dof, DA da3dof, Vec sigNatural, Vec tauNatural,
    std::vector<double> &  sigGlobal, std::vector<double> & gradSigGlobal,
    std::vector<double> & tauGlobal, std::vector<double> & gradTauGlobal,
    std::vector<double> & sigElemental, std::vector<double> & tauElemental) {

  PetscInt Ne;
  int numImages;
  //DA returns the number of nodes.
  //Need the number of elements.
  DAGetInfo(da1dof, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, &numImages, PETSC_NULL,
      PETSC_NULL, PETSC_NULL);
  Ne--;

  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  DAGetCorners(da1dof, &xs, &ys, &zs, &nx, &ny, &nz);

  PetscInt nxe, nye, nze;

  if((xs + nx) == (Ne + 1)) {
    nxe = nx - 1;
  } else {
    nxe = nx;
  }

  if((ys + ny) == (Ne + 1)) {
    nye = ny - 1;
  } else {
    nye = ny;
  }

  if((zs + nz) == (Ne + 1)) {
    nze = nz - 1;
  } else {
    nze = nz;
  }

  Vec sigGvec, tauGvec;
  Vec gradSigGvec, gradTauGvec;

  DACreateGlobalVector(da1dof, &tauGvec);
  DACreateGlobalVector(da3dof, &gradTauGvec);

  VecDuplicate(tauGvec, &sigGvec);
  VecDuplicate(gradTauGvec, &gradSigGvec);

  DANaturalToGlobalBegin(da1dof, tauNatural, INSERT_VALUES, tauGvec);
  DANaturalToGlobalEnd(da1dof, tauNatural, INSERT_VALUES, tauGvec);
  DANaturalToGlobalBegin(da1dof, sigNatural, INSERT_VALUES, sigGvec);
  DANaturalToGlobalEnd(da1dof, sigNatural, INSERT_VALUES, sigGvec);

  PetscScalar**** sigGlobalArr;
  PetscScalar**** tauGlobalArr;
  DAVecGetArrayDOF(da1dof, sigGvec, &sigGlobalArr);
  DAVecGetArrayDOF(da1dof, tauGvec, &tauGlobalArr);

  sigElemental.clear();
  tauElemental.clear();
  for(int k = zs; k < zs + nze; k++) {
    for(int j = ys; j < ys + nye; j++) {
      for(int i = xs; i < xs + nxe; i++) {
        for(int d = 0; d < numImages; d++) {
          double valSig = sigGlobalArr[k][j][i][d];
          double valTau = tauGlobalArr[k][j][i][d];
          sigElemental.push_back(valSig);
          tauElemental.push_back(valTau);
        }
      }
    }
  }

  DAVecRestoreArrayDOF(da1dof, sigGvec, &sigGlobalArr);
  DAVecRestoreArrayDOF(da1dof, tauGvec, &tauGlobalArr);

  computeFDgradient(da1dof, da3dof, sigGvec, gradSigGvec);
  computeFDgradient(da1dof, da3dof, tauGvec, gradTauGvec);

  PetscInt gVecSz;
  VecGetLocalSize(sigGvec, &gVecSz);

  assert(gVecSz == (numImages*nx*ny*nz));

  sigGlobal.resize(numImages*nx*ny*nz);
  gradSigGlobal.resize(3*numImages*nx*ny*nz);
  tauGlobal.resize(numImages*nx*ny*nz);
  gradTauGlobal.resize(3*numImages*nx*ny*nz);

  PetscScalar* sigGlinArr;
  PetscScalar* tauGlinArr;
  PetscScalar* gradSigGlinArr;
  PetscScalar* gradTauGlinArr;

  VecGetArray(sigGvec, &sigGlinArr);
  VecGetArray(gradSigGvec, &gradSigGlinArr);

  VecGetArray(tauGvec, &tauGlinArr);
  VecGetArray(gradTauGvec, &gradTauGlinArr);

  for(int i = 0; i < (numImages*nx*ny*nz); i++) {
    sigGlobal[i] = sigGlinArr[i];
    tauGlobal[i] = tauGlinArr[i];
    for(int j = 0; j < 3; j++) {
      gradSigGlobal[(3*i) + j] = gradSigGlinArr[(3*i) + j];
      gradTauGlobal[(3*i) + j] = gradTauGlinArr[(3*i) + j];
    }
  }

  VecRestoreArray(sigGvec, &sigGlinArr);
  VecRestoreArray(gradSigGvec, &gradSigGlinArr);

  VecRestoreArray(tauGvec, &tauGlinArr);
  VecRestoreArray(gradTauGvec, &gradTauGlinArr);

  VecDestroy(sigGvec);
  VecDestroy(gradSigGvec);

  VecDestroy(tauGvec);
  VecDestroy(gradTauGvec);

}

void createImgNodalNatural(DA da, const std::vector<double>& sigElemImg,
    const std::vector<double>& tauElemImg, Vec & sigNatural, Vec & tauNatural) {

  DACreateNaturalVector(da, &tauNatural);
  VecDuplicate(tauNatural, &sigNatural);

  Vec sigGlobal;
  Vec tauGlobal;
  DACreateGlobalVector(da, &tauGlobal);
  VecDuplicate(tauGlobal, &sigGlobal);

  int numImages;

  //Number of nodes is 1 more than the number of elements
  PetscInt Ne;
  DAGetInfo(da, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,	
      &numImages, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  Ne--;

  VecZeroEntries(sigGlobal);
  VecZeroEntries(tauGlobal);

  PetscScalar**** sigGlobalArr;
  PetscScalar**** tauGlobalArr;
  DAVecGetArrayDOF(da, sigGlobal, &sigGlobalArr);
  DAVecGetArrayDOF(da, tauGlobal, &tauGlobalArr);

  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  PetscInt nxe, nye, nze;

  if((xs + nx) == (Ne + 1)) {
    nxe = nx - 1;
  } else {
    nxe = nx;
  }

  if((ys + ny) == (Ne + 1)) {
    nye = ny - 1;
  } else {
    nye = ny;
  }

  if((zs + nz) == (Ne + 1)) {
    nze = nz - 1;
  } else {
    nze = nz;
  }

  assert(sigElemImg.size() == (numImages*nxe*nye*nze));
  assert(tauElemImg.size() == (numImages*nxe*nye*nze));

  for(int k = zs, ptCnt = 0; k < zs + nze; k++) {
    for(int j = ys; j < ys + nye; j++) {
      for(int i = xs; i < xs + nxe; i++, ptCnt++) {
        for(int d = 0; d < numImages; d++) {
          sigGlobalArr[k][j][i][d] = sigElemImg[(ptCnt*numImages) + d];
          tauGlobalArr[k][j][i][d] = tauElemImg[(ptCnt*numImages) + d];
        }//end for d
      }//end for i
    }//end for j
  }//end for k

  DAVecRestoreArrayDOF(da, sigGlobal, &sigGlobalArr);
  DAVecRestoreArrayDOF(da, tauGlobal, &tauGlobalArr);

  DAGlobalToNaturalBegin(da, tauGlobal, INSERT_VALUES, tauNatural);
  DAGlobalToNaturalEnd(da, tauGlobal, INSERT_VALUES, tauNatural);

  DAGlobalToNaturalBegin(da, sigGlobal, INSERT_VALUES, sigNatural);
  DAGlobalToNaturalEnd(da, sigGlobal, INSERT_VALUES, sigNatural);

  VecDestroy(sigGlobal);
  VecDestroy(tauGlobal);

}


void createImgN0ToNatural(DA da, Vec sigN0, Vec tauN0,
    Vec & sigNatural, Vec & tauNatural, MPI_Comm comm) {

  DACreateNaturalVector(da, &tauNatural);
  VecDuplicate(tauNatural, &sigNatural);

  int* sendSz = NULL;
  int* recvSz = NULL;
  int* sendOff = NULL;
  int* recvOff = NULL;

  PetscInt inSz, outSz;
  VecGetLocalSize(sigN0, &inSz);
  VecGetLocalSize(sigNatural, &outSz);

  //Should actually use MPI_Scatter for optimal performance,
  //but it's okay for now.
  ot::scatterValues(sigN0, sigNatural, inSz, outSz, sendSz, 
      sendOff, recvSz, recvOff, comm);

  ot::scatterValues(tauN0, tauNatural, inSz, outSz, sendSz, 
      sendOff, recvSz, recvOff, comm);

  assert(sendSz != NULL);
  delete [] sendSz;
  sendSz = NULL;

  assert(recvSz != NULL);
  delete [] recvSz;
  recvSz = NULL;

  assert(sendOff != NULL);
  delete [] sendOff;
  sendOff = NULL;

  assert(recvOff != NULL);
  delete [] recvOff;
  recvOff = NULL;
}

void createSeqNodalImageVec(int Ne, int numImages, int rank, int npes,
    const std::vector<double>& img, Vec & imgN0, MPI_Comm comm) {

  VecCreate(comm, &imgN0);

  if(!rank) {
    VecSetSizes(imgN0, numImages*(Ne + 1)*(Ne + 1)*(Ne + 1), PETSC_DECIDE);

    if(npes == 1) {
      VecSetType(imgN0, VECSEQ); 
    } else {
      VecSetType(imgN0, VECMPI); 
    }

    PetscScalar* imgArr;
    VecGetArray(imgN0, &imgArr);

    assert((img.size()) == (numImages*Ne*Ne*Ne));

    for(int k = 0 ; k < Ne; k++) {
      for(int j = 0 ; j < Ne; j++) {
        for(int i = 0 ; i < Ne; i++) {
          int ptIdxN = (((k*(Ne + 1)) + j)*(Ne + 1)) + i;
          int ptIdxE = (((k*Ne) + j)*Ne) + i;
          for(int d = 0; d < numImages; d++) {
            imgArr[(ptIdxN*numImages) + d] = img[(ptIdxE*numImages) + d];
          }//for d
        }//for i
      }//for j
    }//for k

    for(int k = 0 ; k < Ne; k++) {
      for(int j = 0 ; j < Ne; j++) {
        for(int d = 0; d < numImages; d++) {
          imgArr[(((((k*(Ne + 1)) + j)*(Ne + 1)) + Ne)*numImages) + d] = imgArr[(((((k*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1)*numImages) + d];
        }
      }
    }

    for(int k = 0 ; k < Ne; k++) {
      for(int i = 0 ; i < Ne; i++) {
        for(int d = 0; d < numImages; d++) {
          imgArr[(((((k*(Ne + 1)) + Ne)*(Ne + 1)) + i)*numImages) + d] = imgArr[(((((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i)*numImages) + d];
        }
      }
    }

    for(int j = 0 ; j < Ne; j++) {
      for(int i = 0 ; i < Ne; i++) {
        for(int d = 0; d < numImages; d++) {
          imgArr[(((((Ne*(Ne + 1)) + j)*(Ne + 1)) + i)*numImages) + d] = imgArr[((((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + i)*numImages) + d];
        }
      }
    }

    for(int i = 0 ; i < Ne; i++) {
      for(int d = 0; d < numImages; d++) {
        imgArr[(((((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + i)*numImages) + d] = imgArr[((((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i)*numImages) + d];
      }
    }

    for(int j = 0 ; j < Ne; j++) {
      for(int d = 0; d < numImages; d++) {
        imgArr[(((((Ne*(Ne + 1)) + j)*(Ne + 1)) + Ne)*numImages) + d] = imgArr[((((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1)*numImages) + d];
      }
    }

    for(int k = 0 ; k < Ne; k++) {
      for(int d = 0; d < numImages; d++) {
        imgArr[(((((k*(Ne + 1)) + Ne)*(Ne + 1)) + Ne)*numImages) + d] = imgArr[(((((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1)*numImages) + d];
      }
    }

    for(int d = 0; d < numImages; d++) {
      imgArr[(((((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + Ne)*numImages) + d] = imgArr[((((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1)*numImages) + d];
    }

    VecRestoreArray(imgN0, &imgArr);
  } else {
    VecSetSizes(imgN0, 0, PETSC_DECIDE);
    VecSetType(imgN0, VECMPI); 
  }//end if p0
}//end fn.


double gaussNewton(ot::DAMG* damg, double fTol, double xTol, int maxIterCnt, Vec Uin, Vec Uout){

  PetscLogEventBegin(optEvent, 0, 0, 0, 0);

  int iterCnt = 0;

  PetscTruth useSmartLS;
  PetscOptionsHasName(0, "-useSmartLS", &useSmartLS);

  PetscTruth useSteepestDescent;
  PetscOptionsHasName(0, "-useSteepestDescent", &useSteepestDescent);

  PetscTruth saveGradient;
  PetscOptionsHasName(0, "-saveGradient", &saveGradient);

  assert(damg != NULL);

  int nlevels = damg[0]->nlevels;

  assert(damg[nlevels - 1] != NULL);

  ot::DA* daFinest = damg[nlevels - 1]->da;

  assert(daFinest != NULL);

  int rank = daFinest->getRankAll();
  int npes = daFinest->getNpesAll();

  HessData* ctxFinest = static_cast<HessData*>(damg[nlevels - 1]->user);

  assert(ctxFinest != NULL);

  int numGpts = ctxFinest->numGpts;
  double* gPts = ctxFinest->gPts;
  double* gWts = ctxFinest->gWts;
  double mu = ctxFinest->mu;
  double lambda = ctxFinest->lambda;
  double alpha = ctxFinest->alpha;
  int Ne = ctxFinest->Ne;
  int padding = ctxFinest->padding;
  int numImages = ctxFinest->numImages;
  std::vector<std::vector<double> >* tauLocal = ctxFinest->tauLocal;
  std::vector<std::vector<double> >* gradTauLocal = ctxFinest->gradTauLocal;
  std::vector<double>* sigVals = ctxFinest->sigVals;
  std::vector<double>* tauAtU = ctxFinest->tauAtU;
  double****** PhiMatStencil = ctxFinest->PhiMatStencil;
  double**** LaplacianStencil = ctxFinest->LaplacianStencil;
  double**** GradDivStencil = ctxFinest->GradDivStencil;
  unsigned char* bdyArr = ctxFinest->bdyArr;
  Vec uTmp = ctxFinest->uTmp;

  double objVal = 0;
  double objValInit = 0;

  if(!rank) {
    std::cout<<"Updating Hess Contexts..."<<std::endl;
  }

  //1. update HessContext
  updateHessContexts(damg, Uin);

  if(!rank) {
    std::cout<<"Setting KSP..."<<std::endl;
  }

  //2. Set new operators
  if(!useSteepestDescent) {
    ot::DAMGSetKSP(damg, createHessMat,  computeHessMat, computeRHS);
  }

  PetscInt kspMaxIts = 30;
  PetscReal kspRtolInit = 1e-2;
  PetscReal kspAtol = 1e-14;

  PetscOptionsGetInt(0, "-ksp_max_it", &kspMaxIts, 0);
  PetscOptionsGetReal(0, "-ksp_rtol", &kspRtolInit, 0);
  PetscOptionsGetReal(0, "-ksp_atol", &kspAtol, 0);

  PetscReal kspRtol = kspRtolInit;

  if(!useSteepestDescent) {
    assert(DAMGGetKSP(damg) != NULL);
    KSPSetTolerances(DAMGGetKSP(damg), kspRtol, kspAtol, PETSC_DEFAULT, kspMaxIts);
  }

  if(!rank) {
    std::cout<<"Computing Objective..."<<std::endl;
  }

  assert(sigVals != NULL);
  assert(tauAtU != NULL);

  //3. Initial Objective
  objValInit = evalObjective(daFinest, (*sigVals), (*tauAtU), numImages, numGpts, gWts,
      bdyArr, LaplacianStencil, GradDivStencil, mu, lambda, alpha, Uin, uTmp);

  if(!rank) {
    std::cout<<"Computing Gradient..."<<std::endl;
  }

  //4. Initial Gradient
  PetscReal gNormInit, gNorm;
  computeRHS(damg[nlevels - 1], DAMGGetRHS(damg));
  VecNorm(DAMGGetRHS(damg), NORM_2, &gNormInit);

  if(!rank) {
    std::cout<<"Initial: "<<objValInit<<", "<<gNormInit<<std::endl;
  }

  //Line Search Factor
  PetscScalar lsFac = 2.0;

  while(iterCnt < maxIterCnt) {
    iterCnt++;

    //5. Compute Newton step
    if(useSteepestDescent) {
      computeRHS(damg[nlevels - 1], DAMGGetx(damg));
      VecNorm(DAMGGetx(damg), NORM_2, &gNorm);
      VecScale(DAMGGetx(damg), (1.0/gNorm));
    } else {
      ot::DAMGSolve(damg);
      PetscReal finalResNorm;
      KSPGetResidualNorm(DAMGGetKSP(damg), &finalResNorm);
      VecNorm(DAMGGetRHS(damg), NORM_2, &gNorm);
      if(!rank) {
        std::cout<<"Final Res Norm: "<<finalResNorm<<
          " Initial Res Norm: "<<gNorm<<std::endl;
      }
      if(finalResNorm > (0.1*gNorm) ) {
        if(!rank) {
          std::cout<<"Using Steepest Descent Step."<<std::endl;
        }
        VecCopy(DAMGGetRHS(damg), DAMGGetx(damg));
        VecScale(DAMGGetx(damg), (1.0/gNorm));
      }
    }

    //6. Enforce BC
    enforceBC(daFinest, bdyArr, DAMGGetx(damg));

    //Line Search
    if(!useSmartLS) {
      lsFac = 2.0;
    }

    PetscReal stepNormMax;
    VecNorm(DAMGGetx(damg), NORM_INFINITY, &stepNormMax);

    if(gNorm < (fTol*gNormInit)) {
      if(!rank) {
        std::cout<<"GN Exit Type 1"<<std::endl;
        std::cout<<"gNorm: "<<gNorm<<" gNormInit: "<<gNormInit<<std::endl;
      }
      break;
    }

    if(!useSteepestDescent) {
      PetscScalar dirDer;
      VecTDot(DAMGGetx(damg), DAMGGetRHS(damg), &dirDer);
      assert(dirDer > 0.0);
      if(!rank) {
        std::cout<<"Directional Derivative: "<<(-dirDer)<<std::endl;
      }
    }

    unsigned int lsIterCnt = 0;
    do {
      lsFac = 0.5*lsFac;

      //7. Use Uout as tmp vector for line search
      VecWAXPY(Uout, lsFac, DAMGGetx(damg), Uin);

      std::vector<double>  tauAtUtmp;

      assert(tauLocal != NULL);
      assert(gradTauLocal != NULL);

      //8. Interpolate at Tmp Point
      computeTauAtU(daFinest, (*tauLocal), (*gradTauLocal),
          Ne, padding, numImages, Uout,
          PhiMatStencil, numGpts, gPts, tauAtUtmp);

      assert(sigVals != NULL);

      //9.  Objective at tmp Point
      objVal = evalObjective(daFinest, (*sigVals), tauAtUtmp, numImages, numGpts, gWts,
          bdyArr, LaplacianStencil, GradDivStencil, mu, lambda, alpha, Uout, uTmp);

      lsIterCnt++;

    } while( ((lsFac*stepNormMax) >= xTol) && (objVal > objValInit) );

    //11. Update solution
    if(objVal < objValInit) {
      VecAXPY(Uin, lsFac, DAMGGetx(damg));
      char tmpFname[256];
      sprintf(tmpFname, "tmpDisp%d_%d.dat", iterCnt, rank);
      //saveVector(Uin, tmpFname);
    }

    if((lsFac*stepNormMax) < xTol) {
      if(!rank) {
        std::cout<<"GN Exit Type 2"<<std::endl;
        std::cout<<"lsFac: "<<lsFac<<" stepNormMax: "<<stepNormMax<<std::endl;
      }
      break;
    }

    if( objVal < (fTol*objValInit) ) {
      if(!rank) {
        std::cout<<"GN Exit Type 3"<<std::endl;
        std::cout<<"objVal: "<<objVal<<" objValInit: "<<objValInit<<std::endl; 
      }
      break;
    }

    //Prepare for next iteration...
    objValInit = objVal;

    //12. Update Hess Context 
    updateHessContexts(damg, Uin);

    //13. Set New Operators
    if(!useSteepestDescent) {
      ot::DAMGSetKSP(damg, createHessMat,  computeHessMat, computeRHS);
    }

    kspRtol = kspRtolInit*(gNorm/gNormInit);

    if(!useSteepestDescent) {
      KSPSetTolerances(DAMGGetKSP(damg), kspRtol, kspAtol, PETSC_DEFAULT, kspMaxIts);
    }

    //14. Display
    if(!rank) {
      std::cout<<iterCnt<<", "<<objVal<<", "<<gNorm<<", "<<(lsFac*stepNormMax)<<", "<<lsFac<<std::endl;
    }

    if(useSmartLS) {
      if(lsIterCnt == 1) {
        //Accelerate if no LS was performed
        lsFac = 4.0*lsFac;
      }
      if(lsFac > 2.0) {
        lsFac = 2.0;
      }
    }//end if smartLS

  }//end while

  if(!rank) {
    std::cout<<"Final: "<<iterCnt<<", "<<objVal<<", "<<gNorm<<std::endl;
  }

  //Copy solution to output vector
  VecCopy(Uin, Uout);

  if(saveGradient) {
    char fname[256];
    sprintf(fname,"gradJ_%d_%d.dat", rank, npes);
    saveVector(DAMGGetRHS(damg), fname);
  }

  PetscLogEventEnd(optEvent, 0, 0, 0, 0);
  return objVal;
}

double evalGauss3D(double x0, double y0, double z0,
    double sx, double sy, double sz,
    double x, double y, double z)  {

  double expArg = -( ((SQR(x - x0))/(2.0*(SQR(sx)))) + 
      ((SQR(y - y0))/(2.0*(SQR(sy)))) +
      ((SQR(z - z0))/(2.0*(SQR(sz)))) );

  double fVal = exp(expArg);

  return fVal;

}

void coarsenPrlImage(DA daf, DA dac, bool cActive, int numImages,
    const std::vector<double>& imgF, std::vector<double>& imgC) {

  //The processors for dac must be a subset of the processors for daf
  //This function must be called by all processors in daf
  //The zero processor on daf must be cActive and must be
  //the zero processor on dac

  MPI_Comm commF;
  PetscObjectGetComm((PetscObject)daf, &commF);

  int rank;
  MPI_Comm_rank(commF, &rank);

  //Number of nodes is 1 more than the number of elements
  PetscInt Nfe; 
  DAGetInfo(daf, PETSC_NULL, &Nfe, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,	
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  Nfe--;

  double hf = 1.0/static_cast<double>(Nfe);

  PetscInt fxs, fys, fzs;
  PetscInt fnx, fny, fnz;
  DAGetCorners(daf, &fxs, &fys, &fzs, &fnx, &fny, &fnz);

  PetscInt fnxe, fnye, fnze;

  if((fxs + fnx) == (Nfe + 1)) {
    fnxe = fnx - 1;
  } else {
    fnxe = fnx;
  }

  if((fys + fny) == (Nfe + 1)) {
    fnye = fny - 1;
  } else {
    fnye = fny;
  }

  if((fzs + fnz) == (Nfe + 1)) {
    fnze = fnz - 1;
  } else {
    fnze = fnz;
  }

  assert(imgF.size() == (numImages*fnxe*fnye*fnze));

  int npesF;
  MPI_Comm_size(commF, &npesF);

  //First Align imgF with dac partition

  PetscInt cnpx, cnpy, cnpz;
  if(!rank) {
    DAGetInfo(dac, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &cnpx, &cnpy, &cnpz,	
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  }

  par::Mpi_Bcast<PetscInt>(&cnpx, 1, 0, commF);
  par::Mpi_Bcast<PetscInt>(&cnpy, 1, 0, commF);
  par::Mpi_Bcast<PetscInt>(&cnpz, 1, 0, commF);

  PetscInt* clx = new PetscInt[cnpx];
  assert(clx);
  PetscInt* cly = new PetscInt[cnpy];
  assert(cly);
  PetscInt* clz = new PetscInt[cnpz];
  assert(clz);

  if(!rank) {
    const PetscInt* _clx = NULL;
    const PetscInt* _cly = NULL;
    const PetscInt* _clz = NULL;

    DAGetOwnershipRanges(dac, &_clx, &_cly, &_clz);

    for(int i = 0; i < cnpx; i++) {
      clx[i] = _clx[i];
    }

    for(int i = 0; i < cnpy; i++) {
      cly[i] = _cly[i];
    }

    for(int i = 0; i < cnpz; i++) {
      clz[i] = _clz[i];
    }
  }

  par::Mpi_Bcast<PetscInt>(clx, cnpx, 0, commF);
  par::Mpi_Bcast<PetscInt>(cly, cnpy, 0, commF);
  par::Mpi_Bcast<PetscInt>(clz, cnpz, 0, commF);

  std::vector<double> scanLx(cnpx);
  std::vector<double> scanLy(cnpy);
  std::vector<double> scanLz(cnpz);

  PetscInt Nce = Nfe/2;
  double hc = 1.0/static_cast<double>(Nce);

  assert(!(scanLx.empty()));
  assert(!(scanLy.empty()));
  assert(!(scanLz.empty()));

  scanLx[0] = 0;
  scanLy[0] = 0;
  scanLz[0] = 0;
  for(int i = 1; i < cnpx; i++) {
    scanLx[i] = scanLx[i - 1] + (static_cast<double>(clx[i - 1])*hc);
  }
  for(int i = 1; i < cnpy; i++) {
    scanLy[i] = scanLy[i - 1] + (static_cast<double>(cly[i - 1])*hc);
  }
  for(int i = 1; i < cnpz; i++) {
    scanLz[i] = scanLz[i - 1] + (static_cast<double>(clz[i - 1])*hc);
  }

  delete[] clx;
  delete[] cly;
  delete[] clz;
  clx = NULL;
  cly = NULL;
  clz = NULL;

  int* sendCnts = new int[npesF];
  assert(sendCnts);

  int* part = new int[(fnxe*fnye*fnze)];
  assert(part);

  for(int i = 0; i < npesF; i++) {
    sendCnts[i] = 0;
  }//end for i

  for(int k = fzs, ptCnt = 0; k < (fzs + fnze); k++) {
    for(int j = fys; j < (fys + fnye); j++) {
      for(int i = fxs; i < (fxs + fnxe); i++, ptCnt++) {
        double xPt = (static_cast<double>(i))*hf;
        double yPt = (static_cast<double>(j))*hf;
        double zPt = (static_cast<double>(k))*hf;
        unsigned int xRes, yRes, zRes;
        seq::maxLowerBound<double>(scanLx, xPt, xRes, 0, 0);
        seq::maxLowerBound<double>(scanLy, yPt, yRes, 0, 0);
        seq::maxLowerBound<double>(scanLz, zPt, zRes, 0, 0);
        part[ptCnt] = (((zRes*cnpy) + yRes)*cnpx) + xRes;
        assert(part[ptCnt] < npesF);
        sendCnts[part[ptCnt]] += (3 + numImages);
      }//end for i
    }//end for j
  }//end for k

  scanLx.clear();
  scanLy.clear();
  scanLz.clear();  

  int* recvCnts = new int[npesF];
  assert(recvCnts);

  par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, commF);

  int* sendOffsets = new int[npesF];
  assert(sendOffsets);

  int* recvOffsets = new int[npesF];
  assert(recvOffsets);

  sendOffsets[0] = 0;
  recvOffsets[0] = 0;
  for(int i = 1; i < npesF; i++) {
    sendOffsets[i] = sendOffsets[i - 1] + sendCnts[i - 1];
    recvOffsets[i] = recvOffsets[i - 1] + recvCnts[i - 1];
  }//end for i

  int* tmpSendCnts = new int[npesF];  
  assert(tmpSendCnts);
  for(int i = 0; i < npesF; i++) {
    tmpSendCnts[i] = 0;
  }

  double* xyzSendVals = new double[(3 + numImages)*(fnxe*fnye*fnze)];
  assert(xyzSendVals);

  for(int k = fzs, ptCnt = 0; k < (fzs + fnze); k++) {
    for(int j = fys; j < (fys + fnye); j++) {
      for(int i = fxs; i < (fxs + fnxe); i++, ptCnt++) {
        double xPt = (static_cast<double>(i))*hf;
        double yPt = (static_cast<double>(j))*hf;
        double zPt = (static_cast<double>(k))*hf;
        int idx = sendOffsets[part[ptCnt]] + tmpSendCnts[part[ptCnt]];
        tmpSendCnts[part[ptCnt]] += (3 + numImages);
        xyzSendVals[idx] = xPt;
        xyzSendVals[idx + 1] = yPt;
        xyzSendVals[idx + 2] = zPt;
        for(int d = 0; d < numImages; d++) {
          xyzSendVals[idx + 3 + d] = imgF[(ptCnt*numImages) + d];
        }//end for d
      }//end for i
    }//end for j
  }//end for k

  assert(tmpSendCnts);
  delete [] tmpSendCnts;
  tmpSendCnts = NULL;

  assert(part);
  delete [] part;
  part = NULL;

  double* xyzRecvVals = new double[recvOffsets[npesF - 1] + recvCnts[npesF - 1]];
  assert(xyzRecvVals);

  par::Mpi_Alltoallv_sparse<double>(xyzSendVals, sendCnts, sendOffsets,
      xyzRecvVals, recvCnts, recvOffsets, commF);

  int numRecvPts = (recvOffsets[npesF - 1] + recvCnts[npesF - 1])/(3 + numImages);

  assert(xyzSendVals);
  delete [] xyzSendVals;
  xyzSendVals = NULL;

  assert(recvCnts);
  delete [] recvCnts;
  recvCnts = NULL;

  assert(sendCnts);
  delete [] sendCnts;
  sendCnts = NULL;

  assert(recvOffsets);
  delete [] recvOffsets;
  recvOffsets = NULL;

  assert(sendOffsets);
  delete [] sendOffsets;
  sendOffsets = NULL;

  PetscInt cnxe = 0;
  PetscInt cnye = 0;
  PetscInt cnze = 0;

  PetscInt cxs, cys, czs;
  if(cActive) {
    PetscInt cnx, cny, cnz;
    DAGetCorners(dac, &cxs, &cys, &czs, &cnx, &cny, &cnz);
    if((cxs + cnx) == (Nce + 1)) {
      cnxe = cnx - 1;
    } else {
      cnxe = cnx;
    }

    if((cys + cny) == (Nce + 1)) {
      cnye = cny - 1;
    } else {
      cnye = cny;
    }

    if((czs + cnz) == (Nce + 1)) {
      cnze = cnz - 1;
    } else {
      cnze = cnz;
    }
  }//end if active

  assert(numRecvPts == (8*cnxe*cnye*cnze));

  std::vector<double> imgFlocal(numImages*8*cnxe*cnye*cnze);

  for(int i = 0; i < numRecvPts; i++) {
    double xPt = xyzRecvVals[((3 + numImages)*i)];
    double yPt = xyzRecvVals[((3 + numImages)*i) + 1];
    double zPt = xyzRecvVals[((3 + numImages)*i) + 2];
    int xi = (static_cast<int>(xPt/hf)) - (2*cxs);
    int yi = (static_cast<int>(yPt/hf)) - (2*cys);
    int zi = (static_cast<int>(zPt/hf)) - (2*czs);
    int idx = (((zi*(2*cnye)) + yi)*(2*cnxe)) + xi;
    for(int d = 0; d < numImages; d++) {
      imgFlocal[(idx*numImages) + d] = xyzRecvVals[((3 + numImages)*i) + 3 + d];
    }//end for d
  }//end for i

  assert(xyzRecvVals);
  delete [] xyzRecvVals;
  xyzRecvVals = NULL;

  imgC.resize(numImages*cnxe*cnye*cnze);

  //Coarsen Locally
  for(int k = 0; k < cnze; k++) {
    for(int j = 0; j < cnye; j++) {
      for(int i = 0; i < cnxe; i++) {
        for(int d = 0; d < numImages; d++) {
          imgC[(((((k*cnye) + j)*cnxe) + i)*numImages) + d] = 0.125*(
              imgFlocal[((((((2*k)*(2*cnye)) + (2*j))*(2*cnxe)) + (2*i))*numImages) + d] +
              imgFlocal[((((((2*k)*(2*cnye)) + (2*j))*(2*cnxe)) + ((2*i) + 1))*numImages) + d] +
              imgFlocal[((((((2*k)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + (2*i))*numImages) + d] +
              imgFlocal[((((((2*k)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + ((2*i) + 1))*numImages) + d] +
              imgFlocal[(((((((2*k) + 1)*(2*cnye)) + (2*j))*(2*cnxe)) + (2*i))*numImages) + d] +
              imgFlocal[(((((((2*k) + 1)*(2*cnye)) + (2*j))*(2*cnxe)) + ((2*i) + 1))*numImages) + d] +
              imgFlocal[(((((((2*k) + 1)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + (2*i))*numImages) + d] +
              imgFlocal[(((((((2*k) + 1)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + ((2*i) + 1))*numImages) + d]);
        }//end for d
      }//end for i
    }//end for j
  }//end for k

}


void coarsenImage(int Nfe, int numImages, const std::vector<double>& imgF,
    std::vector<double>& imgC) {

  int Nce = Nfe/2;

  imgC.resize(numImages*Nce*Nce*Nce);

  assert(imgF.size() == (numImages*Nfe*Nfe*Nfe));

  for(int k = 0; k < Nce; k++) {
    for(int j = 0; j < Nce; j++) {
      for(int i = 0; i < Nce; i++) {
        assert( ((((((2*k) + 1)*Nfe) + ((2*j) + 1))*Nfe) + ((2*i) + 1)) < (Nfe*Nfe*Nfe) );
        for(int d = 0; d < numImages; d++) {
          imgC[(((((k*Nce) + j)*Nce) + i)*numImages) + d] = 0.125*(
              imgF[((((((2*k)*Nfe) + (2*j))*Nfe) + (2*i))*numImages) + d] +
              imgF[((((((2*k)*Nfe) + (2*j))*Nfe) + ((2*i) + 1))*numImages) + d] +
              imgF[((((((2*k)*Nfe) + ((2*j) + 1))*Nfe) + (2*i))*numImages) + d] +
              imgF[((((((2*k)*Nfe) + ((2*j) + 1))*Nfe) + ((2*i) + 1))*numImages) + d] +
              imgF[(((((((2*k) + 1)*Nfe) + (2*j))*Nfe) + (2*i))*numImages) + d] +
              imgF[(((((((2*k) + 1)*Nfe) + (2*j))*Nfe) + ((2*i) + 1))*numImages) + d] +
              imgF[(((((((2*k) + 1)*Nfe) + ((2*j) + 1))*Nfe) + (2*i))*numImages) + d] +
              imgF[(((((((2*k) + 1)*Nfe) + ((2*j) + 1))*Nfe) + ((2*i) + 1))*numImages) + d]);
        }
      }
    }
  }
}


PetscErrorCode computeRHS(ot::DAMG damg, Vec rhs) {
  PetscFunctionBegin;

  PetscLogEventBegin(evalGradEvent, 0, 0, 0, 0);

  //This function computes the negative of the gradient

  assert(damg != NULL);

  ot::DA* da = damg->da;
  HessData* data = static_cast<HessData*>(damg->user);

  assert(da != NULL);
  assert(data != NULL);

  unsigned char* bdyArr = data->bdyArr;
  double mu = data->mu;
  double lambda = data->lambda;
  double alpha = data->alpha;
  int numImages = data->numImages;
  int numGpts = data->numGpts;
  double* gWts = data->gWts;
  std::vector<double>* sigVals = data->sigVals;
  std::vector<double>* tauAtU = data->tauAtU; 
  std::vector<double>* gradTauAtU = data->gradTauAtU; 
  double****** PhiMatStencil = data->PhiMatStencil;
  double**** LaplacianStencil = data->LaplacianStencil;
  double**** GradDivStencil = data->GradDivStencil;
  Vec U = data->U;
  Vec uTmp = data->uTmp;

  elasMatVec(da, bdyArr, LaplacianStencil, GradDivStencil,  mu, lambda, U, uTmp);

  assert(sigVals != NULL);
  assert(tauAtU != NULL);
  assert(gradTauAtU != NULL);

  computeGradientImgPart(da, (*sigVals), (*tauAtU), (*gradTauAtU),
      bdyArr, PhiMatStencil, numImages, numGpts, gWts, rhs);

  VecAXPY(rhs, -alpha, uTmp);

  enforceBC(da, bdyArr, rhs);

  PetscLogEventEnd(evalGradEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}


double evalObjective(ot::DA* da, const std::vector<double> & sigVals,
    const std::vector<double> & tauAtU, int numImages, int numGpts, double* gWts,
    unsigned char* bdyArr, double**** LaplacianStencil, 
    double**** GradDivStencil, double mu, double lambda, double alpha, Vec U, Vec tmp) {

  PetscLogEventBegin(evalObjEvent, 0, 0, 0, 0);

  elasMatVec(da, bdyArr, LaplacianStencil, GradDivStencil, mu, lambda, U, tmp);

  double objImgPart = computeObjImgPart(da, sigVals, tauAtU, numImages, numGpts, gWts); 

  PetscScalar objElasPart; 
  VecTDot(U, tmp, &objElasPart);

  PetscLogEventEnd(evalObjEvent, 0, 0, 0, 0);

  return (0.5*(objImgPart + (alpha*objElasPart)));
}

double computeObjImgPart(ot::DA* da, const std::vector<double>& sigVals,
    const std::vector<double> & tauAtU, int numImages, int numGpts, double* gWts) {

  double objImgPartLocal = 0.0;

  assert(da != NULL);

  if(da->iAmActive()) {
    unsigned int maxD = da->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);
    unsigned int ptsCtr = 0;
    for(da->init<ot::DA_FLAGS::WRITABLE>();
        da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
        da->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = da->getCurrentOffset();
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      double hOct = hFac*(static_cast<double>(1u << (maxD - lev)));
      double x0 = hFac*static_cast<double>(xint);
      double y0 = hFac*static_cast<double>(yint);
      double z0 = hFac*static_cast<double>(zint);
      double elemIntFac = (hOct*hOct*hOct)/8.0;
      double elemInt = 0.0;
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            double localVal = 0.0;
            for(int i = 0; i < numImages; i++) {
              localVal += ((sigVals[(ptsCtr*numImages) + i] - tauAtU[(ptsCtr*numImages) + i])*
                  (sigVals[(ptsCtr*numImages) + i] - tauAtU[(ptsCtr*numImages) + i]));
            }//end for i
            elemInt += (gWts[m]*gWts[n]*gWts[p]*localVal);
            ptsCtr++;
          }//end for p
        }//end for n
      }//end for m
      objImgPartLocal += (elemIntFac*elemInt);
    }//end WRITABLE
  }//end if active

  double objImgPartGlobal;

  par::Mpi_Allreduce<double>(&objImgPartLocal, &objImgPartGlobal,
      1, MPI_SUM, da->getComm());

  return objImgPartGlobal;
}

void computeGradientImgPart(ot::DA* da, const std::vector<double> & sigVals,
    const std::vector<double> & tauAtU, const std::vector<double> & gradTauAtU, 
    unsigned char* bdyArr, double****** PhiMatStencil,
    int numImages, int numGpts, double* gWts, Vec g) {

  VecZeroEntries(g);

  if(da->iAmActive()) {
    PetscScalar* gArr;
    da->vecGetBuffer(g, gArr, false, false, false, 3);

    unsigned int maxD = da->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);
    unsigned int elemCtr = 0;
    for(da->init<ot::DA_FLAGS::WRITABLE>();
        da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
        da->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = da->getCurrentOffset();
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      double hOct = hFac*(static_cast<double>(1u << (maxD - lev)));
      double x0 = hFac*static_cast<double>(xint);
      double y0 = hFac*static_cast<double>(yint);
      double z0 = hFac*static_cast<double>(zint);
      double elemIntFac = (hOct*hOct*hOct)/8.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType, hnMask, childNum)
        for(int m = 0; m < numGpts; m++) {
          for(int n = 0; n < numGpts; n++) {
            for(int p = 0; p < numGpts; p++) {
              double localVal[3];
              for(int dof = 0; dof < 3; dof++) {
                localVal[dof] = 0.0;
              }
              unsigned int ptIdx = (((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p;
              for(int i = 0; i < numImages; i++) {
                for(int dof = 0; dof < 3; dof++) {
                  localVal[dof] += ((sigVals[(ptIdx*numImages) + i] - tauAtU[(ptIdx*numImages) + i])*
                      gradTauAtU[(((ptIdx*numImages) + i)*3) + dof]);
                }
              }//end for i
              for(int k = 0; k < 8; k++) {
                if(!bdyArr[indices[k]]) {
                  double phiVal = PhiMatStencil[childNum][elemType][k][m][n][p];
                  for(int dof = 0; dof < 3; dof++) {
                    gArr[(3*indices[k]) + dof] += (elemIntFac*gWts[m]*gWts[n]*gWts[p]*localVal[dof]*phiVal);
                  }
                }//end if boundary
              }//end for k
            }//end for p
          }//end for n
        }//end for m
      elemCtr++;
    }//end WRITABLE

    da->WriteToGhostsBegin<PetscScalar>(gArr, 3);
    da->WriteToGhostsEnd<PetscScalar>(gArr, 3);

    da->vecRestoreBuffer(g, gArr, false, false, false, 3);
  }//end if active

}


void destroyHessContexts(ot::DAMG* damg) {
  assert(damg);
  int nlevels = damg[0]->nlevels; 
  for(int i = 0; i < nlevels; i++) {
    assert(damg[i] != NULL);
    HessData* ctx = static_cast<HessData*>(damg[i]->user);
    assert(ctx != NULL);
    if(ctx->bdyArr) {
      delete [] (ctx->bdyArr);
      ctx->bdyArr = NULL;
    }
    if(ctx->gtVec) {
      delete (ctx->gtVec);
      ctx->gtVec = NULL;  
    }
    if(ctx->sigVals) {
      delete (ctx->sigVals);
      ctx->sigVals = NULL;  
    }
    if(ctx->tauLocal) {
      delete (ctx->tauLocal);
      ctx->tauLocal = NULL;  
    }
    if(ctx->gradTauLocal) {
      delete (ctx->gradTauLocal);
      ctx->gradTauLocal = NULL;  
    }
    if(ctx->tauAtU) {
      delete (ctx->tauAtU);
      ctx->tauAtU = NULL;  
    }
    if(ctx->gradTauAtU) {
      delete (ctx->gradTauAtU);
      ctx->gradTauAtU = NULL;  
    }
    if(ctx->uTmp) {
      VecDestroy(ctx->uTmp);
      ctx->uTmp = NULL;
    }
    if(ctx->Jmat_private) {
      MatDestroy(ctx->Jmat_private);
      ctx->Jmat_private = NULL;
    }
    if(ctx->inTmp) {
      VecDestroy(ctx->inTmp);
      ctx->inTmp = NULL;
    }
    if(ctx->outTmp) {
      VecDestroy(ctx->outTmp);
      ctx->outTmp = NULL;
    }
    delete ctx;
    ctx = NULL;
  }
}

void updateHessContexts(ot::DAMG* damg, Vec U) {

  PetscLogEventBegin(updateHessContextEvent, U, 0, 0, 0);

  assert(damg != NULL);

  int nlevels = damg[0]->nlevels;
  HessData* ctxFinest = static_cast<HessData*>(damg[nlevels - 1]->user);
  ot::DA* daFinest = damg[nlevels - 1]->da;

  assert(ctxFinest != NULL);
  assert(daFinest != NULL);

  ctxFinest->U = U;
  double****** PhiMatStencil = ctxFinest->PhiMatStencil;
  int Ne = ctxFinest->Ne;
  int padding = ctxFinest->padding;
  int numImages = ctxFinest->numImages;
  int numGpts = ctxFinest->numGpts;
  double* gPts = ctxFinest->gPts;
  std::vector<std::vector<double> >* tauLocal = ctxFinest->tauLocal;
  std::vector<std::vector<double> >* gradTauLocal = ctxFinest->gradTauLocal;
  std::vector<double>* tauAtU = ctxFinest->tauAtU;
  std::vector<double>* gradTauAtU = ctxFinest->gradTauAtU;

  assert(tauLocal != NULL);
  assert(gradTauLocal != NULL);
  assert(tauAtU != NULL);
  assert(gradTauAtU != NULL);

  //Compute the finest vector first
  std::vector<double> nodalGradTauAtU;
  computeNodalGradTauAtU(daFinest, (*tauLocal), (*gradTauLocal), 
      Ne, padding, numImages, U, nodalGradTauAtU);

  computeTauAtU(daFinest, (*tauLocal), (*gradTauLocal),
      Ne, padding, numImages, U, PhiMatStencil, numGpts, gPts, (*tauAtU));

  computeGradTauAtU(daFinest, (*tauLocal), (*gradTauLocal),
      Ne, padding, numImages, U, PhiMatStencil, numGpts, gPts, (*gradTauAtU));

  std::vector<double>* gtVecFinest = ctxFinest->gtVec;

  assert(gtVecFinest != NULL);

  daFinest->createVector<double>((*gtVecFinest), false, false, 6);

  for(int i = 0; i < gtVecFinest->size(); i++) {
    (*gtVecFinest)[i] = 0;
  }

  int sixMap[][3] = { {0, 1, 2},
    {-1000, 3, 4},
    {-1000, -1000, 5} };

  unsigned int finestNodeSize = daFinest->getNodeSize();

  for(int i = 0; i < finestNodeSize; i++) {    
    for(int dof1 = 0; dof1 < 3; dof1++) {
      for(int dof2 = dof1; dof2 < 3; dof2++) {
        for(int j = 0; j < numImages; j++) {
          (*gtVecFinest)[(6*i) + (sixMap[dof1][dof2])] += 
            (nodalGradTauAtU[(((i*numImages) + j)*3) + dof1]*
             nodalGradTauAtU[(((i*numImages) + j)*3) + dof2]);
        }//end for j
      }//end for dof2
    }//end for dof1
  }//end for i

  //Coarsen using injection
  for(int i = (nlevels - 1); i > 0; i--) {
    HessData* ctxF = (static_cast<HessData*>(damg[i]->user));    
    HessData* ctxC = (static_cast<HessData*>(damg[i - 1]->user));
    assert(ctxF != NULL);
    assert(ctxC != NULL);
    ot::DA* dac = damg[i - 1]->da;
    ot::DA* daf = NULL;
    std::vector<double>* cVec = ctxC->gtVec;
    std::vector<double>* fVec = NULL;

    if(damg[i]->da_aux) {
      daf = damg[i]->da_aux;
      fVec = new std::vector<double>; 
      assert( (ctxF->gtVec) != NULL );
      par::scatterValues<double>((*(ctxF->gtVec)), (*fVec), 
          (6*(daf->getNodeSize())), damg[0]->comm);
    } else {
      daf = damg[i]->da;
      fVec = ctxF->gtVec;
    } 

    assert(fVec != NULL);
    assert(cVec != NULL);

    ot::injectNodalVector<double>(dac, daf, 6, (*fVec), (*cVec), zeroDouble);

    if(damg[i]->da_aux) {
      delete fVec;
      fVec = NULL;
    }
  }//end for i

  PetscLogEventEnd(updateHessContextEvent, U, 0, 0, 0);

}


void createHessContexts(ot::DAMG* damg, int Ne, int padding, int numImages,
    const std::vector<std::vector<double> >& sigLocal,
    const std::vector<std::vector<double> >& gradSigLocal,
    const std::vector<std::vector<double> >& tauLocal, 
    const std::vector<std::vector<double> >& gradTauLocal,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts, double mu, double lambda, double alpha) {

  PetscLogEventBegin(createHessContextEvent, 0, 0, 0, 0);

  assert(damg != NULL);

  int nlevels = damg[0]->nlevels; 

  for(int i = 0; i < nlevels; i++) {
    HessData* ctx = new HessData;
    assert(ctx);
    ctx->gtVec = new std::vector<double>;
    ctx->tauLocal = NULL;
    ctx->gradTauLocal = NULL;
    ctx->sigVals = NULL;
    ctx->tauAtU = NULL;
    ctx->gradTauAtU = NULL;
    ctx->bdyArr = NULL; 
    ctx->Jmat_private = NULL;
    ctx->inTmp = NULL;
    ctx->outTmp = NULL;
    ctx->U = NULL;
    ctx->uTmp = NULL;
    ctx->Ne = Ne;
    ctx->padding = padding;
    ctx->numImages = numImages;
    ctx->mu = mu;
    ctx->lambda = lambda;
    ctx->alpha = alpha;
    ctx->numGpts = numGpts;
    ctx->gWts = gWts;
    ctx->gPts = gPts;
    ctx->PhiMatStencil = PhiMatStencil;
    ctx->LaplacianStencil = LaplacianStencil;
    ctx->GradDivStencil = GradDivStencil;

    if(i == (nlevels - 1)) {
      //Finest level
      damg[i]->da->createVector(ctx->uTmp, false, false, 3);
      ctx->tauLocal = new std::vector<std::vector<double> >; 
      ctx->gradTauLocal = new std::vector<std::vector<double> >; 
      ctx->sigVals = new std::vector<double>;
      ctx->tauAtU = new std::vector<double>; 
      ctx->gradTauAtU = new std::vector<double >; 

      computeSigVals(damg[i]->da, sigLocal, gradSigLocal,
          Ne, padding, numImages, numGpts, gPts, (*(ctx->sigVals)));

      //We can avoid this copy, but then we must trust the user to not change
      //tau and gradTau for the entire run.
      (*(ctx->tauLocal)) = tauLocal;
      (*(ctx->gradTauLocal)) = gradTauLocal;
    }

    std::vector<unsigned char> tmpBdyFlags;
    std::vector<unsigned char> tmpBdyFlagsAux;
    unsigned char* bdyArrAux = NULL;

    //This will create a nodal, non-ghosted, 1 dof array
    assignBoundaryFlags(damg[i]->da, tmpBdyFlags);
    ((damg[i])->da)->vecGetBuffer<unsigned char>(tmpBdyFlags, ctx->bdyArr, false, false, true, 1);
    if(damg[i]->da->iAmActive()) {
      ((damg[i])->da)->ReadFromGhostsBegin<unsigned char>(ctx->bdyArr, 1);
    }

    if(damg[i]->da_aux) {
      assignBoundaryFlags( damg[i]->da_aux, tmpBdyFlagsAux);
      ((damg[i])->da_aux)->vecGetBuffer<unsigned char>(tmpBdyFlagsAux, bdyArrAux, false, false, true, 1);
      if(damg[i]->da_aux->iAmActive()) {
        ((damg[i])->da_aux)->ReadFromGhostsBegin<unsigned char>(bdyArrAux, 1);
      }
    }

    tmpBdyFlags.clear();
    tmpBdyFlagsAux.clear();

    if(damg[i]->da->iAmActive()) {
      ((damg[i])->da)->ReadFromGhostsEnd<unsigned char>(ctx->bdyArr);
    }

    if((damg[i])->da_aux) {
      if(damg[i]->da_aux->iAmActive()) {
        ((damg[i])->da_aux)->ReadFromGhostsEnd<unsigned char>(bdyArrAux);
      }
    }

    for(int loopCtr = 0; loopCtr < 2; loopCtr++) {
      ot::DA* da = NULL;
      unsigned char* suppressedDOFptr = NULL;
      unsigned char* bdyArrPtr = NULL;
      if(loopCtr == 0) {
        da = damg[i]->da;
        suppressedDOFptr = damg[i]->suppressedDOF;
        bdyArrPtr = ctx->bdyArr;
      } else {
        da = damg[i]->da_aux;
        suppressedDOFptr = damg[i]->suppressedDOFaux;
        bdyArrPtr = bdyArrAux;
      }
      if(da) {
        if(da->iAmActive()) {
          for(da->init<ot::DA_FLAGS::ALL>(); 
              da->curr() < da->end<ot::DA_FLAGS::ALL>();
              da->next<ot::DA_FLAGS::ALL>()) {
            unsigned int indices[8];
            da->getNodeIndices(indices);
            for(unsigned int k = 0; k < 8; k++) {
              for(unsigned int d = 0; d < 3; d++) {
                suppressedDOFptr[(3*indices[k]) + d] = bdyArrPtr[indices[k]];
              }
            }
          }//end ALL
        }
      }
    }

    if(bdyArrAux) {
      delete [] bdyArrAux;
      bdyArrAux = NULL;
    }

    (damg[i])->user = ctx;
  }//end for i

  PetscLogEventEnd(createHessContextEvent, 0, 0, 0, 0);

}
void concatDispFile(char *elasDisp,int numProc, char *genDisp){

  Vec v1;
  if (numProc==1){
    char fname1[256];
    sprintf(fname1,"%s_%d_%d.dat",elasDisp,0,numProc);
    loadSeqVector(v1, fname1);
    PetscInt vlen1;
    VecGetLocalSize(v1, &vlen1);
    PetscScalar* v1Arr;
    VecGetArray(v1, &v1Arr);
    FILE* fptr = fopen(genDisp, "wb");
    PetscInt vlen = vlen1 ;
    fwrite(&vlen, sizeof(PetscInt), 1, fptr);
    fwrite(v1Arr, sizeof(PetscScalar), vlen1, fptr);
    VecRestoreArray(v1, &v1Arr);
    fclose(fptr);
  }   
  else if (numProc >1){
    
    char fname1[256],fname2[256];
    sprintf(fname1,"%s_%d_%d.dat",elasDisp,0,numProc);
    sprintf(fname2,"%s_%d_%d.dat",elasDisp,1,numProc);
    concatDisp(fname1,fname2,genDisp);   
    for (int i=2;i <numProc;i ++){
      sprintf(fname1,"%s_%d_%d.dat",elasDisp,i,numProc);
      concatDisp(genDisp,fname1,genDisp); 
    } 


  }
}


void concatDisp(char *dispFile1,char *dispFile2,char *dispTarget){
   Vec v1, v2;
  loadSeqVector(v1, dispFile1);
  loadSeqVector(v2, dispFile2);

  PetscInt vlen1;
  VecGetLocalSize(v1, &vlen1);
  PetscInt vlen2;
  VecGetLocalSize(v2, &vlen2);

  PetscScalar* v1Arr;
  VecGetArray(v1, &v1Arr);
  PetscScalar* v2Arr;
  VecGetArray(v2, &v2Arr);
 
  FILE* fptr = fopen(dispTarget, "wb");

 
  PetscInt vlen = vlen1 + vlen2;
  fwrite(&vlen, sizeof(PetscInt), 1, fptr);
  fwrite(v1Arr, sizeof(PetscScalar), vlen1, fptr);
  fwrite(v2Arr, sizeof(PetscScalar), vlen2, fptr);
 
  VecRestoreArray(v1, &v1Arr);
  VecRestoreArray(v2, &v2Arr);
  fclose(fptr);
  VecDestroy(v1);
  VecDestroy(v2);

}


/* Concatenate v1 and v2 and save */
void concatSaveVector(Vec v1, Vec v2, char* fname) {
  FILE* fptr = fopen(fname, "wb");

  PetscInt vlen1;
  VecGetLocalSize(v1, &vlen1);
  PetscInt vlen2;
  VecGetLocalSize(v2, &vlen2);

  PetscScalar* v1Arr;
  VecGetArray(v1, &v1Arr);
  PetscScalar* v2Arr;
  VecGetArray(v2, &v2Arr);
  
  PetscInt vlen = vlen1 + vlen2;
  fwrite(&vlen, sizeof(PetscInt), 1, fptr);
  fwrite(v1Arr, sizeof(PetscScalar), vlen1, fptr);
  fwrite(v2Arr, sizeof(PetscScalar), vlen2, fptr);
 /* 
  for (int i = 0; i < vlen; i++){
	 fprintf(fptr, "%le ", vArr[i]);
  }
  */
  //printf("VLEN = %d %d %d\n", vlen, vlen1, vlen2);

  VecRestoreArray(v1, &v1Arr);
  VecRestoreArray(v2, &v2Arr);
  fclose(fptr);
}

void writeScalarImage(char* fnamePrefix, int nx, int ny, int nz, std::vector<double> & img, int imgId) {
  struct dsr hdr;

  memset(&hdr, 0, sizeof(struct dsr));
  for(int i = 0; i < 8; i++) {
    hdr.dime.pixdim[i] = 0.0;
  }

  hdr.dime.vox_offset = 0.0;
  hdr.dime.funused1 = 0.0;
  hdr.dime.funused2 = 0.0;
  hdr.dime.funused3 = 0.0;
  hdr.dime.cal_max = 0.0;
  hdr.dime.cal_min = 0.0;

  //DOUBLE
  hdr.dime.datatype = 64;
  hdr.dime.bitpix = 64;

  char fname[256];
  sprintf(fname, "%s.hdr", fnamePrefix);
  FILE *fp = fopen(fname, "wb");

  hdr.dime.dim[0] = 4; /* all Analyze images are taken as 4 dimensional */
  hdr.hk.regular = 'r';
  hdr.hk.sizeof_hdr = sizeof(struct dsr);

  hdr.dime.dim[1] = nx; /* slice width in pixels */
  hdr.dime.dim[2] = ny; /* slice height in pixels */
  hdr.dime.dim[3] = nz; /* volume depth in slices */
  hdr.dime.dim[4] = 1; /* number of volumes per file */

  int maxVal, minVal;
  maxVal = img[0];
  minVal = img[0];
  int totalPixels = nx*ny*nz;
  for(int i = 0; i < totalPixels; i++) {
    if(maxVal < img[i]) {
      maxVal = img[i];
    }
    if(minVal > img[i]) {
      minVal = img[i];
    }
  }

  hdr.dime.glmax = maxVal; /* maximum voxel value */
  hdr.dime.glmin = minVal; /* minimum voxel value */

  /* Set the voxel dimension fields:
     A value of 0.0 for these fields implies that the value is unknown.
     Change these values to what is appropriate for your data
     or pass additional command line arguments */

  hdr.dime.pixdim[1] = 0.0; /* voxel x dimension */
  hdr.dime.pixdim[2] = 0.0; /* voxel y dimension */
  hdr.dime.pixdim[3] = 0.0; /* pixel z dimension, slice thickness */

  /* Assume zero offset in .img file, byte at which pixel
     data starts in the image file */

  hdr.dime.vox_offset = 0.0;

  /* Planar Orientation; */
  /* Movie flag OFF: 0 = transverse, 1 = coronal, 2 = sagittal
     Movie flag ON: 3 = transverse, 4 = coronal, 5 = sagittal */

  hdr.hist.orient = 0;

  /* up to 3 characters for the voxels units label; i.e. mm., um., cm. */ 

  strcpy(hdr.dime.vox_units," ");

  /* up to 7 characters for the calibration units label; i.e. HU */

  strcpy(hdr.dime.cal_units," ");

  /* Calibration maximum and minimum values;
     values of 0.0 for both fields imply that no
     calibration max and min values are used */

  hdr.dime.cal_max = 0.0;
  hdr.dime.cal_min = 0.0;

  fwrite(&hdr,sizeof(struct dsr),1,fp);

  fclose(fp);

  char fnameHdr[256];
  sprintf(fnameHdr,"%s_%d.hdr",fnamePrefix,imgId);
  fp = fopen(fnameHdr,"wb");
  fwrite(&hdr,sizeof(struct dsr),1,fp);
  fclose(fp);

  sprintf(fname, "%s_%d.img", fnamePrefix,imgId);
  fp = fopen(fname, "wb");

  assert(!(img.empty()));

  fwrite((&(*(img.begin()))), sizeof(double), totalPixels, fp);
  fclose(fp);
}

void writeVecImg(char *scalarImg,char *vecImg,int numImg)
{
//  PetscInt numImg=6;


  struct dsr hdr;
  
  std::vector<double> scalarI,vecI[numImg];
  readScalarImage(scalarImg,&hdr,scalarI);
  int imgSize= scalarI.size();
   enum tissueI {skull=0,ventCSF=10,CSF=50,GM=150,WM=250,TM=200};
 
 
//initialize vecI to zero
  for (int i=0;i<numImg;i++) vecI[i].resize(imgSize);

  for (int i=0;i<imgSize;i++){
    if (scalarI[i]==skull) vecI[0][i]=1;
    else if (scalarI[i]==ventCSF) vecI[1][i]=1;
    else if (scalarI[i]==CSF) vecI[2][i]=1;
    else if (scalarI[i]==GM) vecI[3][i]=1;
    else if (scalarI[i]==WM) vecI[4][i]=1;
    else if (scalarI[i]==TM) vecI[5][i]=1;
  }

 int dimX=hdr.dime.dim[1];
 int dimY=hdr.dime.dim[2];
 int dimZ=hdr.dime.dim[3];

 for (int i=0;i<numImg;i++){
    char vecImgFile[256];
//    sprintf(vecImgFile,"%s_%d",vecImg,i);
    writeScalarImage(vecImg,dimX,dimY,dimZ,vecI[i],i);
    
  } 
 
}

void writeScalarImg(char *vecImg, char *scalarImg, int numImg)
{
//  PetscInt numImg=6;
//  PetscOptionsGetInt(0,"-numImages", &numImg,0);
  struct dsr hdr;  

  std::vector<double> scalarI,vecI[numImg];

  for (int i=0;i<numImg;i++){
    char vecImgFile[256];
//    sprintf(vecImgFile,"%s_%d",vecImg,i);
    readScalarImage(vecImg,&hdr,vecI[i],i);
    
  }
 int dimX=hdr.dime.dim[1];
 int dimY=hdr.dime.dim[2];
 int dimZ=hdr.dime.dim[3];

 int imgSize=dimX*dimY*dimZ;

 double tissueI[6]={0,10,50,150,250,200};
  scalarI.resize(imgSize);
  double maxI;
  int indexMax;
  for (int i=0;i<imgSize;i++){
    maxI=vecI[0][i];
    indexMax=0;
    for (int j=1;j<numImg;j++){
      if (vecI[j][i]> maxI){
        maxI = vecI[j][i];
        indexMax = j;
      }
      
    }
    scalarI[i] = tissueI[indexMax];

  } 
  int imgId=0;
  writeScalarImage(scalarImg,dimX,dimY,dimZ,scalarI,imgId);

}



void writeImage(char* fnamePrefix, int imgId, int nx, int ny, int nz, std::vector<double> & img) {
  struct dsr hdr;

  memset(&hdr, 0, sizeof(struct dsr));
  for(int i = 0; i < 8; i++) {
    hdr.dime.pixdim[i] = 0.0;
  }

  hdr.dime.vox_offset = 0.0;
  hdr.dime.funused1 = 0.0;
  hdr.dime.funused2 = 0.0;
  hdr.dime.funused3 = 0.0;
  hdr.dime.cal_max = 0.0;
  hdr.dime.cal_min = 0.0;

  //DOUBLE
  hdr.dime.datatype = 64;
  hdr.dime.bitpix = 64;

  char fname[256];
  sprintf(fname, "%s.hdr", fnamePrefix);
  FILE *fp = fopen(fname, "wb");

  hdr.dime.dim[0] = 4; /* all Analyze images are taken as 4 dimensional */
  hdr.hk.regular = 'r';
  hdr.hk.sizeof_hdr = sizeof(struct dsr);

  hdr.dime.dim[1] = nx; /* slice width in pixels */
  hdr.dime.dim[2] = ny; /* slice height in pixels */
  hdr.dime.dim[3] = nz; /* volume depth in slices */
  hdr.dime.dim[4] = 1; /* number of volumes per file */

  int maxVal, minVal;
  int totalPixels = nx*ny*nz;
  assert(img.size() == totalPixels);

  maxVal = static_cast<int>(img[0]);
  minVal = static_cast<int>(img[0]);
  for(int i = 0; i < totalPixels; i++) {
    if(maxVal < img[i]) {
      maxVal = static_cast<int>(img[i]);
    }
    if(minVal > img[i]) {
      minVal = static_cast<int>(img[i]);
    }
  }

  hdr.dime.glmax = maxVal; /* maximum voxel value */
  hdr.dime.glmin = minVal; /* minimum voxel value */

  /* Set the voxel dimension fields:
     A value of 0.0 for these fields implies that the value is unknown.
     Change these values to what is appropriate for your data
     or pass additional command line arguments */

  hdr.dime.pixdim[1] = 0.0; /* voxel x dimension */
  hdr.dime.pixdim[2] = 0.0; /* voxel y dimension */
  hdr.dime.pixdim[3] = 0.0; /* pixel z dimension, slice thickness */

  /* Assume zero offset in .img file, byte at which pixel
     data starts in the image file */

  hdr.dime.vox_offset = 0.0;

  /* Planar Orientation; */
  /* Movie flag OFF: 0 = transverse, 1 = coronal, 2 = sagittal
     Movie flag ON: 3 = transverse, 4 = coronal, 5 = sagittal */

  hdr.hist.orient = 0;

  /* up to 3 characters for the voxels units label; i.e. mm., um., cm. */ 

  strcpy(hdr.dime.vox_units," ");

  /* up to 7 characters for the calibration units label; i.e. HU */

  strcpy(hdr.dime.cal_units," ");

  /* Calibration maximum and minimum values;
     values of 0.0 for both fields imply that no
     calibration max and min values are used */

  hdr.dime.cal_max = 0.0;
  hdr.dime.cal_min = 0.0;

  fwrite(&hdr,sizeof(struct dsr),1,fp);

  fclose(fp);

  sprintf(fname, "%s_%d.img", fnamePrefix, imgId);
  fp = fopen(fname, "wb");

  assert(!(img.empty()));

  fwrite((&(*(img.begin()))), sizeof(double), totalPixels, fp);
  fclose(fp);
}

void readImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> &  img)
{
  char fname[256];

  sprintf(fname, "%s.hdr", fnamePrefix);

  FILE *fp = fopen(fname, "rb");

  fread(hdr, sizeof(struct dsr), 1, fp);

  fclose(fp); 

  int dimX = hdr->dime.dim[1];
  int dimY = hdr->dime.dim[2];
  int dimZ = hdr->dime.dim[3];

  img.resize(dimX*dimY*dimZ);

  sprintf(fname, "%s.img", fnamePrefix);

  fp = fopen(fname, "rb");

  int totalPixels = (dimX*dimY*dimZ); 
  int pixByte = (hdr->dime.bitpix/8);
  int rawSize = totalPixels*pixByte;
  char* imgPtr = new char[rawSize];
  assert(imgPtr);

  fread(imgPtr, rawSize, 1, fp);

  switch(hdr->dime.datatype) {
    case DT_UNSIGNED_CHAR: {
                             for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                               unsigned char* v = (unsigned char*)(imgPtr + (pxCnt*pixByte));
                               img[pxCnt] = static_cast<double>(*v);
                             }
                             break;
                           }
    case DT_SIGNED_SHORT: {
                            for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                              short int* v = (short int*)(imgPtr + (pxCnt*pixByte));
                              img[pxCnt] = static_cast<double>(*v);
                            }
                            break;
                          }
    case DT_SIGNED_INT: {
                          for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                            int* v = (int*)(imgPtr + (pxCnt*pixByte));
                            img[pxCnt] = static_cast<double>(*v);
                          }
                          break;
                        }
    case DT_FLOAT: {
                     for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                       float* v = (float*)(imgPtr + (pxCnt*pixByte));
                       img[pxCnt] = static_cast<double>(*v);
                     }
                     break;
                   }
    case DT_DOUBLE: {
                      for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                        double* v = (double*)(imgPtr + (pxCnt*pixByte));
                        img[pxCnt] = static_cast<double>(*v);
                      }
                      break;
                    }
    default: {
               std::cout<<"This format is not supported ."<<std::endl;
               assert(false);
             }
  }

  delete [] imgPtr;
  imgPtr = NULL;

  fclose(fp); 
}


void readScalarImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> &  img, int imgId)
{
  char fname[256];

  sprintf(fname, "%s.hdr", fnamePrefix);

  FILE *fp = fopen(fname, "rb");

  fread(hdr, sizeof(struct dsr), 1, fp);

  fclose(fp); 

  int dimX = hdr->dime.dim[1];
  int dimY = hdr->dime.dim[2];
  int dimZ = hdr->dime.dim[3];

  img.resize(dimX*dimY*dimZ);

  sprintf(fname, "%s_%d.img", fnamePrefix,imgId);

  fp = fopen(fname, "rb");

  int totalPixels = (dimX*dimY*dimZ); 
  int pixByte = (hdr->dime.bitpix/8);
  int rawSize = totalPixels*pixByte;
  char* imgPtr = new char[rawSize];
  assert(imgPtr);

  fread(imgPtr, rawSize, 1, fp);

  switch(hdr->dime.datatype) {
    case DT_UNSIGNED_CHAR: {
                             for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                               unsigned char* v = (unsigned char*)(imgPtr + (pxCnt*pixByte));
                               img[pxCnt] = static_cast<double>(*v);
                             }
                             break;
                           }
    case DT_SIGNED_SHORT: {
                            for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                              short int* v = (short int*)(imgPtr + (pxCnt*pixByte));
                              img[pxCnt] = static_cast<double>(*v);
                            }
                            break;
                          }
    case DT_SIGNED_INT: {
                          for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                            int* v = (int*)(imgPtr + (pxCnt*pixByte));
                            img[pxCnt] = static_cast<double>(*v);
                          }
                          break;
                        }
    case DT_FLOAT: {
                     for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                       float* v = (float*)(imgPtr + (pxCnt*pixByte));
                       img[pxCnt] = static_cast<double>(*v);
                     }
                     break;
                   }
    case DT_DOUBLE: {
                      for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                        double* v = (double*)(imgPtr + (pxCnt*pixByte));
                        img[pxCnt] = static_cast<double>(*v);
                      }
                      break;
                    }
    default: {
               std::cout<<"This format is not supported ."<<std::endl;
               assert(false);
             }
  }

  delete [] imgPtr;
  imgPtr = NULL;

  fclose(fp); 
}

void readScalarImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> &  img)
{
  char fname[256];

  sprintf(fname, "%s.hdr", fnamePrefix);

  FILE *fp = fopen(fname, "rb");

  fread(hdr, sizeof(struct dsr), 1, fp);

  fclose(fp); 

  int dimX = hdr->dime.dim[1];
  int dimY = hdr->dime.dim[2];
  int dimZ = hdr->dime.dim[3];

  img.resize(dimX*dimY*dimZ);

  sprintf(fname, "%s.img", fnamePrefix);

  fp = fopen(fname, "rb");

  int totalPixels = (dimX*dimY*dimZ); 
  int pixByte = (hdr->dime.bitpix/8);
  int rawSize = totalPixels*pixByte;
  char* imgPtr = new char[rawSize];
  assert(imgPtr);

  fread(imgPtr, rawSize, 1, fp);

  switch(hdr->dime.datatype) {
    case DT_UNSIGNED_CHAR: {
                             for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                               unsigned char* v = (unsigned char*)(imgPtr + (pxCnt*pixByte));
                               img[pxCnt] = static_cast<double>(*v);
                             }
                             break;
                           }
    case DT_SIGNED_SHORT: {
                            for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                              short int* v = (short int*)(imgPtr + (pxCnt*pixByte));
                              img[pxCnt] = static_cast<double>(*v);
                            }
                            break;
                          }
    case DT_SIGNED_INT: {
                          for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                            int* v = (int*)(imgPtr + (pxCnt*pixByte));
                            img[pxCnt] = static_cast<double>(*v);
                          }
                          break;
                        }
    case DT_FLOAT: {
                     for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                       float* v = (float*)(imgPtr + (pxCnt*pixByte));
                       img[pxCnt] = static_cast<double>(*v);
                     }
                     break;
                   }
    case DT_DOUBLE: {
                      for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                        double* v = (double*)(imgPtr + (pxCnt*pixByte));
                        img[pxCnt] = static_cast<double>(*v);
                      }
                      break;
                    }
    default: {
               std::cout<<"This format is not supported ."<<std::endl;
               assert(false);
             }
  }

  delete [] imgPtr;
  imgPtr = NULL;

  fclose(fp); 
}



void readImages(char* fnamePrefix, struct dsr* hdr, int numImages, std::vector<double> &  img)
{
  char fname[256];

  sprintf(fname, "%s.hdr", fnamePrefix);

  FILE *fp = fopen(fname, "rb");

  fread(hdr, sizeof(struct dsr), 1, fp);

  fclose(fp); 

  int dimX = hdr->dime.dim[1];
  int dimY = hdr->dime.dim[2];
  int dimZ = hdr->dime.dim[3];

  int totalPixels = (dimX*dimY*dimZ); 
  int pixByte = (hdr->dime.bitpix/8);
  int rawSize = totalPixels*pixByte;
  char* imgPtr = new char[rawSize];
  assert(imgPtr);

  img.resize(numImages*dimX*dimY*dimZ);

  for(int imgId = 0; imgId < numImages; imgId++) {
    sprintf(fname, "%s_%d.img", fnamePrefix, imgId);

    fp = fopen(fname, "rb");

    fread(imgPtr, rawSize, 1, fp);

    switch(hdr->dime.datatype) {
      case DT_UNSIGNED_CHAR: {
                               for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                                 unsigned char* v = (unsigned char*)(imgPtr + (pxCnt*pixByte));
                                 img[(pxCnt*numImages) + imgId] = static_cast<double>(*v);
                               }
                               break;
                             }
      case DT_SIGNED_SHORT: {
                              for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                                short int* v = (short int*)(imgPtr + (pxCnt*pixByte));
                                img[(pxCnt*numImages) + imgId] = static_cast<double>(*v);
                              }
                              break;
                            }
      case DT_SIGNED_INT: {
                            for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                              int* v = (int*)(imgPtr + (pxCnt*pixByte));
                              img[(pxCnt*numImages) + imgId] = static_cast<double>(*v);
                            }
                            break;
                          }
      case DT_FLOAT: {
                       for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                         float* v = (float*)(imgPtr + (pxCnt*pixByte));
                         img[(pxCnt*numImages) + imgId] = static_cast<double>(*v);
                       }
                       break;
                     }
      case DT_DOUBLE: {
                        for(int pxCnt = 0; pxCnt < totalPixels; pxCnt++) {
                          double* v = (double*)(imgPtr + (pxCnt*pixByte));
                          img[(pxCnt*numImages) + imgId] = static_cast<double>(*v);
                        }
                        break;
                      }
      default: {
                 std::cout<<"This format is not supported ."<<std::endl;
                 assert(false);
               }
    }

    fclose(fp); 

  }//end for imgId

  delete [] imgPtr;
  imgPtr = NULL;

}


void genDispImg(char *disp,double dispScale, char* templateImg,char *dispImg,int
computeDetJacMat, int numImg)
{

  for (int imgId=0;imgId < numImg;imgId ++ ){

      bool computeDetJac = (bool) computeDetJacMat;
      Vec U;
      loadSeqVector(U, disp) ;
      VecScale(U, dispScale);

      struct dsr hdr;
      std::vector<double> img;
      readScalarImage(templateImg, &hdr, img,imgId);

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
    
      DADestroy(da1dof);
      DADestroy(da3dof);


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
//   int numImages= tau.size();
      
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
    
//      for (int i=0; i <numImages; i++) imgFinalNodal[i].clear();
     imgFinalNodal.clear();
      //Remove out of range values Image
      //Convert to integer values
/*      for(int i = 0; i < imgFinal.size(); i++) {
        if(imgFinal[i] < 0.0) {
          imgFinal[i] = 0.0;
        }
        if(imgFinal[i] > 255.0) {
          imgFinal[i] = 255.0;
        }
        imgFinal[i] = floor(imgFinal[i]);
      }

      std::cout<<"Passed Stage 7"<<std::endl;
*/
//  Check out
      writeScalarImage(dispImg, Ne, Ne, Ne, imgFinal,imgId);
      
      std::cout<<"Passed Stage 8"<<std::endl;

  }


}



void loadSeqVector(Vec & v, char* fname) {
  FILE* fptr = fopen(fname, "rb");

  PetscInt vlen;
  fread(&vlen, sizeof(PetscInt), 1, fptr);

  VecCreate(PETSC_COMM_SELF, &v);
  VecSetSizes(v, vlen, PETSC_DECIDE);
  VecSetType(v, VECSEQ);

  PetscScalar* vArr;
  VecGetArray(v, &vArr);

  fread(vArr, sizeof(PetscScalar), vlen, fptr);

  VecRestoreArray(v, &vArr);

  fclose(fptr);
}


void loadPrlVector(Vec & v, char* fname) {
  FILE* fptr = fopen(fname, "rb");

  PetscInt vlen;
  fread(&vlen, sizeof(PetscInt), 1, fptr);

  VecCreate(MPI_COMM_WORLD, &v);
  VecSetSizes(v, vlen, PETSC_DECIDE);
  VecSetType(v, VECMPI);

  PetscScalar* vArr;
  VecGetArray(v, &vArr);

  fread(vArr, sizeof(PetscScalar), vlen, fptr);

  VecRestoreArray(v, &vArr);

  fclose(fptr);
}


void saveVector(Vec v, char* fname) {
  FILE* fptr = fopen(fname, "wb");

  PetscInt vlen;
  VecGetLocalSize(v, &vlen);

  PetscScalar* vArr;
  VecGetArray(v, &vArr);

  fwrite(&vlen, sizeof(PetscInt), 1, fptr);
  fwrite(vArr, sizeof(PetscScalar), vlen, fptr);

  VecRestoreArray(v, &vArr);

  fclose(fptr);
}


PetscErrorCode computeHessMat(ot::DAMG damg, Mat J, Mat B) {
  //For matShells nothing to be done here.
  PetscFunctionBegin;

  assert(damg != NULL);
  HessData* data = (static_cast<HessData*>(damg->user));

  assert(data != NULL);

  PetscTruth isshell;
  PetscTypeCompare((PetscObject)B, MATSHELL, &isshell);

  assert(J == B);

  if(isshell) {
    if( data->Jmat_private == NULL ) {
      //inactive processors will return
      PetscFunctionReturn(0);
    } else {
      J = data->Jmat_private;
      B = data->Jmat_private;
    }
  }

  PetscTypeCompare((PetscObject)B, MATSHELL, &isshell);

  if(isshell) {
    PetscFunctionReturn(0);
  }

  ot::DA* da = damg->da;
  unsigned int maxD;
  double hFac;

  assert(da != NULL);

  if(da->iAmActive()) {
    maxD = da->getMaxDepth();
    hFac = 1.0/((double)(1u << (maxD-1)));
  }

  unsigned char* bdyArr = data->bdyArr;
  double**** LaplacianStencil = data->LaplacianStencil;
  double**** GradDivStencil = data->GradDivStencil;
  double mu = data->mu;
  double lambda = data->lambda;
  double alpha = data->alpha;

  int sixMap[][3] = { {0, 1, 2},
    {1, 3, 4},
    {2, 4, 5} };

  MatZeroEntries(B);

  double* gtArr = NULL;
  assert( (data->gtVec) != NULL );
  da->vecGetBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);

  std::vector<ot::MatRecord> records;
  if(da->iAmActive()) {

    //We need ghost values because we will only be using a WRITABLE loop.
    //Can overlap communication and computation later
    //We typically build the matrix only for the coarsest level and we typically
    //only use 1 processor for the coarsest level so it is okay anyway.
    da->ReadFromGhostsBegin<double>(gtArr, 6);
    da->ReadFromGhostsEnd<double>(gtArr);

    for(da->init<ot::DA_FLAGS::WRITABLE>();
        da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
        da->next<ot::DA_FLAGS::WRITABLE>()) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      double facImg = (h*h*h)/8.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0; k < 8; k++) {
          /*Avoid Dirichlet Node Rows during ADD_VALUES loop.*/
          /*Need a separate INSERT_VALUES loop for those*/
          if(!(bdyArr[indices[k]])) {
            for(int j = 0; j < 8; j++) {
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  ot::MatRecord currRec;
                  currRec.rowIdx = indices[k];
                  currRec.colIdx = indices[j];
                  currRec.rowDim = dof;
                  currRec.colDim = dof;
                  currRec.val = (alpha*mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]);
                  records.push_back(currRec);
                } /*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    ot::MatRecord currRec;
                    currRec.rowIdx = indices[k];
                    currRec.colIdx = indices[j];
                    currRec.rowDim = dofOut;
                    currRec.colDim = dofIn;
                    currRec.val = (alpha*(mu + lambda)*facElas*
                        GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn]);
                    records.push_back(currRec);
                  } /*end for dofIn*/
                } /*end for dofOut*/
              } /*end if boundary*/
            } /*end for j*/
            for(int dofOut = 0; dofOut < 3; dofOut++) {
              for(int dofIn = 0; dofIn < 3; dofIn++) {
                ot::MatRecord currRec;
                currRec.rowIdx = indices[k];
                currRec.colIdx = indices[k];
                currRec.rowDim = dofOut;
                currRec.colDim = dofIn;
                currRec.val = facImg*gtArr[(6*indices[k]) + sixMap[dofOut][dofIn]];
                records.push_back(currRec);
              } /*end for dofIn*/
            } /*end for dofOut*/
          } /*end if boundary*/
        } /*end for k*/
      if(records.size() > 1000) {
        /*records will be cleared inside the function*/
        da->setValuesInMatrix(B, records, 3, ADD_VALUES);
      }
    } /*end writable*/
    da->setValuesInMatrix(B, records, 3, ADD_VALUES);
  } /*end if active*/

  MatAssemblyBegin(B, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(B, MAT_FLUSH_ASSEMBLY);

  if(da->iAmActive()) {
    /*There will be repetitions here, but it is harmless.*/
    for(da->init<ot::DA_FLAGS::WRITABLE>();
        da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
        da->next<ot::DA_FLAGS::WRITABLE>()) {
      unsigned int indices[8];
      da->getNodeIndices(indices);
      for(int k = 0;k < 8;k++) {
        /*Insert values for Dirichlet Node Rows only.*/
        if(bdyArr[indices[k]]) {
          for(int dof = 0; dof < 3; dof++) {
            ot::MatRecord currRec;
            currRec.rowIdx = indices[k];
            currRec.colIdx = indices[k];
            currRec.rowDim = dof;
            currRec.colDim = dof;
            currRec.val = alpha;
            records.push_back(currRec);
          } /*end for dof*/
        }
      } /*end for k*/
      if(records.size() > 1000) {
        /*records will be cleared inside the function*/
        da->setValuesInMatrix(B, records, 3, INSERT_VALUES);
      }
    } /*end writable*/
    da->setValuesInMatrix(B, records, 3, INSERT_VALUES);
  } /*end if active*/

  MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

  da->vecRestoreBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);

  PetscFunctionReturn(0);
}

PetscErrorCode hessMatMult(Mat J, Vec in, Vec out)
{
  PetscFunctionBegin;

  PetscLogEventBegin(hessMultEvent, in, out, 0, 0);

  ot::DAMG damg;
  MatShellGetContext(J, (void**)(&damg));

  assert(damg != NULL);

  bool isFinestLevel = (damg->nlevels == 1);

  if(isFinestLevel) {
    PetscLogEventBegin(hessFinestMultEvent, in, out, 0, 0);
  }

  ot::DA* da = damg->da;
  assert(da != NULL);

  HessData* data = (static_cast<HessData*>(damg->user));
  assert(data != NULL);

  unsigned char* bdyArr = data->bdyArr;
  double**** LaplacianStencil = data->LaplacianStencil;
  double**** GradDivStencil = data->GradDivStencil;
  double mu = data->mu;
  double lambda = data->lambda;
  double alpha = data->alpha;

  VecZeroEntries(out);

  int sixMap[][3] = { {0, 1, 2},
    {1, 3, 4},
    {2, 4, 5} };

  unsigned int maxD;
  double hFac;
  if(da->iAmActive()) {
    maxD = da->getMaxDepth();
    hFac = 1.0/((double)(1u << (maxD-1)));

    PetscScalar *outArr=NULL;
    PetscScalar *inArr=NULL;
    double* gtArr;

    assert((data->gtVec) != NULL);

    /*Nodal,Non-Ghosted,Read,6 dof*/
    da->vecGetBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);

    /*Nodal,Non-Ghosted,Read,3 dof*/
    da->vecGetBuffer(in, inArr, false, false, true, 3);

    /*Nodal,Non-Ghosted,Write,3 dof*/
    da->vecGetBuffer(out, outArr, false, false, false, 3);

    da->ReadFromGhostsBegin<PetscScalar>(inArr, 3);

    da->ReadFromGhostsEnd<PetscScalar>(inArr);

    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      double facImg = (h*h*h)/8.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(bdyArr[indices[k]]) {
            /*Dirichlet Node Row*/
            for(int dof = 0; dof < 3; dof++) {
              outArr[(3*indices[k]) + dof] =  alpha*inArr[(3*indices[k]) + dof];
            }/*end for dof*/
          } else {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (alpha*mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += (alpha*(mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
            for(int dofOut = 0; dofOut < 3; dofOut++) {
              for(int dofIn = 0; dofIn < 3; dofIn++) {
                outArr[(3*indices[k]) + dofOut] += (facImg*
                    gtArr[(6*indices[k]) + sixMap[dofOut][dofIn]]*
                    inArr[(3*indices[k]) + dofIn]);
              }/*end for dofIn*/
            }/*end for dofOut*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for ALL*/

    da->vecRestoreBuffer(in, inArr, false, false, true, 3);

    da->vecRestoreBuffer(out, outArr, false, false, false, 3);

    da->vecRestoreBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);

  } /*end if active*/

  /*2 IOP = 1 FLOP. Loop counters are included too.*/
  PetscLogFlops(6891*(da->getGhostedElementSize()));

  if(isFinestLevel) {
    PetscLogEventEnd(hessFinestMultEvent, in, out, 0, 0);
  }

  PetscLogEventEnd(hessMultEvent, in, out, 0, 0);

  PetscFunctionReturn(0);
}


PetscErrorCode hessShellMatMult(Mat J, Vec in, Vec out)
{
  PetscFunctionBegin;

  ot::DAMG damg;
  MatShellGetContext(J, (void**)(&damg));

  assert(damg != NULL);

  HessData* ctx = (static_cast<HessData*>(damg->user));
  assert(ctx != NULL);

  if(damg->da->iAmActive()) {      
    PetscScalar* inArray;
    PetscScalar* outArray;

    VecGetArray(in, &inArray);
    VecGetArray(out, &outArray);

    VecPlaceArray(ctx->inTmp, inArray);
    VecPlaceArray(ctx->outTmp, outArray);

    MatMult(ctx->Jmat_private, ctx->inTmp, ctx->outTmp);

    VecResetArray(ctx->inTmp);
    VecResetArray(ctx->outTmp);

    VecRestoreArray(in, &inArray);
    VecRestoreArray(out, &outArray);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode createHessMat(ot::DAMG damg, Mat *jac) {
  PetscFunctionBegin;
  PetscInt buildFullCoarseMat;
  PetscInt buildFullMatAll;
  int totalLevels;
  PetscTruth flg;
  PetscOptionsGetInt(PETSC_NULL,"-buildFullCoarseMat",&buildFullCoarseMat,&flg);
  PetscOptionsGetInt(PETSC_NULL,"-buildFullMatAll",&buildFullMatAll,&flg);
  if(buildFullMatAll) {
    buildFullCoarseMat = 1;
  }
  totalLevels = damg->totalLevels;
  ot::DA* da = damg->da;
  int myRank;
  MPI_Comm_rank(da->getComm(), &myRank);
  if( totalLevels == damg->nlevels ) {
    //This is the coarsest.
    if( (!myRank) && (buildFullCoarseMat) ) {
      std::cout<<"Building Full Hess Mat at the coarsest level."<<std::endl;
    }
    char matType[30];
    if(buildFullCoarseMat) {
      if(!(da->computedLocalToGlobal())) {
        da->computeLocalToGlobalMappings();
      }
      PetscTruth typeFound;
      PetscOptionsGetString(PETSC_NULL,"-fullJacMatType",matType,30,&typeFound);
      if(!typeFound) {
        std::cout<<"I need a MatType for the full matrix!"<<std::endl;
        assert(false);
      }
    }
    bool requirePrivateMats = (da->getNpesActive() != da->getNpesAll());    
    if(requirePrivateMats) {
      unsigned int  m,n;
      m=n=(3*(da->getNodeSize()));
      HessData* ctx = static_cast<HessData*>(damg->user);
      if(da->iAmActive()) {
        if(buildFullCoarseMat) {
          da->createActiveMatrix(ctx->Jmat_private, matType, 3);
        } else {
          MatCreateShell(da->getCommActive(), m ,n, PETSC_DETERMINE,
              PETSC_DETERMINE, damg, (&(ctx->Jmat_private)));

          MatShellSetOperation(ctx->Jmat_private, MATOP_MULT,
              (void (*)(void)) hessMatMult);

          MatShellSetOperation(ctx->Jmat_private, MATOP_DESTROY,
              (void (*)(void)) hessMatDestroy);
        }
        MatGetVecs(ctx->Jmat_private, &(ctx->inTmp), &(ctx->outTmp));
      } else {
        ctx->Jmat_private = NULL;
        ctx->inTmp = NULL;
        ctx->outTmp = NULL;
      }
      //Need a MATShell wrapper anyway. But, the matvecs are not implemented for
      //this matrix. However, a matmult function is required for compute
      //residuals
      MatCreateShell(damg->comm, m ,n, PETSC_DETERMINE, PETSC_DETERMINE, damg, jac);
      MatShellSetOperation(*jac ,MATOP_DESTROY, (void (*)(void)) hessMatDestroy);
      MatShellSetOperation(*jac, MATOP_MULT, (void (*)(void)) hessShellMatMult);
    } else {
      if(buildFullCoarseMat) {
        da->createMatrix(*jac, matType, 3);
      } else {
        unsigned int  m,n;
        m=n=(3*(da->getNodeSize()));
        MatCreateShell(damg->comm, m ,n, PETSC_DETERMINE, PETSC_DETERMINE, damg, jac);
        MatShellSetOperation(*jac ,MATOP_MULT, (void (*)(void)) hessMatMult);
        MatShellSetOperation(*jac ,MATOP_DESTROY, (void (*)(void)) hessMatDestroy);
      }
    }
    if( (!myRank) && (buildFullCoarseMat) ) {
      std::cout<<"Finished Building Full Hess Mat at the coarsest level."<<std::endl;
    }
  }else {  
    //This is not the coarsest level. No need to bother with KSP_Shell
    if(buildFullMatAll) {
      if( !myRank ) {
        std::cout<<"Building Full Hess Mat at level: "<<(damg->nlevels)<<std::endl;
      }
      if(!(da->computedLocalToGlobal())) {
        da->computeLocalToGlobalMappings();
      }
      char matType[30];
      PetscTruth typeFound;
      PetscOptionsGetString(PETSC_NULL,"-fullJacMatType",matType,30,&typeFound);
      if(!typeFound) {
        std::cout<<"I need a MatType for the full matrix!"<<std::endl;
        assert(false);
      }
      da->createMatrix(*jac, matType, 3);
      if(!myRank) {
        std::cout<<"Finished Building Full Hess Mat at level: "<<(damg->nlevels)<<std::endl;
      }
    } else {
      //Create a MATShell
      //The size this processor owns ( without ghosts).
      unsigned int  m,n;
      m=n=(3*(da->getNodeSize()));
      MatCreateShell(damg->comm, m ,n, PETSC_DETERMINE, PETSC_DETERMINE, damg, jac);
      MatShellSetOperation(*jac ,MATOP_MULT, (void (*)(void)) hessMatMult);
      MatShellSetOperation(*jac ,MATOP_DESTROY, (void (*)(void)) hessMatDestroy);
    }
  }
  PetscFunctionReturn(0);
}//end fn.


PetscErrorCode hessMatDestroy(Mat J) {
  PetscFunctionBegin;

  //Since the context must be destroyed before DAMGDestroy, which calls
  //MatDestroy, we will not destroy anything here 
  PetscFunctionReturn(0);
}


PetscErrorCode elasMatVec(ot::DA* da, unsigned char* bdyArr,
    double**** LaplacianStencil, double**** GradDivStencil,
    double mu, double lambda, Vec in, Vec out) {
  PetscFunctionBegin;

  PetscLogEventBegin(elasMultEvent, 0, 0, 0, 0);

  VecZeroEntries(out);

  assert(da != NULL);

  unsigned int maxD;
  double hFac;
  if(da->iAmActive()) {
    maxD = da->getMaxDepth();
    hFac = 1.0/((double)(1u << (maxD-1)));

    PetscScalar *outArr=NULL;
    PetscScalar *inArr=NULL;

    /*Nodal,Non-Ghosted,Read,3 dof*/
    da->vecGetBuffer(in, inArr, false, false, true, 3);

    /*Nodal,Non-Ghosted,Write,3 dof*/
    da->vecGetBuffer(out, outArr, false, false, false, 3);

    da->ReadFromGhostsBegin<PetscScalar>(inArr, 3);
    da->ReadFromGhostsEnd<PetscScalar>(inArr);

    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(!bdyArr[indices[k]]) {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += ((mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for ALL*/

    da->vecRestoreBuffer(in, inArr, false, false, true, 3);

    da->vecRestoreBuffer(out, outArr, false, false, false, 3);

  } /*end if active*/

  PetscLogEventEnd(elasMultEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}


void mortonToRgGlobalDisp(ot::DAMG* damg, DA dar, int Ne,
    std::vector<double> & dispOct, Vec & UrgGlobal) {

  PetscInt xs, ys, zs, nx, ny, nz;
  DAGetCorners(dar, &xs, &ys, &zs, &nx, &ny, &nz);

  int nxe = nx;
  int nye = ny;
  int nze = nz;
  if((xs + nx) == (Ne + 1)) {
    nxe = nx - 1;
  } 
  if((ys + ny) == (Ne + 1)) {
    nye = ny - 1;
  } 
  if((zs + nz) == (Ne + 1)) {
    nze = nz - 1;
  } 

  double h = 1.0/static_cast<double>(Ne);
  std::vector<double> pts;
  for(int k = zs; k < (zs + nze); k++) {
    for(int j = ys; j < (ys + nye); j++) {
      for(int i = xs; i < (xs + nxe); i++) {
        pts.push_back(static_cast<double>(i)*h);
        pts.push_back(static_cast<double>(j)*h);
        pts.push_back(static_cast<double>(k)*h);
      }
    }
  }

  assert(damg != NULL);
  assert(DAMGGetDA(damg) != NULL);

  std::vector<double> dispRg;
  ot::interpolateData(DAMGGetDA(damg), dispOct, dispRg, NULL, 3, pts);
  pts.clear();

  DACreateGlobalVector(dar, &UrgGlobal);

  VecZeroEntries(UrgGlobal);

  PetscScalar**** globalArr;
  DAVecGetArrayDOF(dar, UrgGlobal, &globalArr);

  assert((dispRg.size()) == (3*nxe*nye*nze));

  int ptCnt = 0;
  for(int k = zs; k < (zs + nze); k++) {
    for(int j = ys; j < (ys + nye); j++) {
      for(int i = xs; i < (xs + nxe); i++) {
        for(int d = 0; d < 3; d++) {
          assert( ((3*ptCnt) + d) < dispRg.size() );
          globalArr[k][j][i][d] = dispRg[(3*ptCnt) + d];
        }//end for d
        ptCnt++;
      }//end for i
    }//end for j
  }//end for k

  DAVecRestoreArrayDOF(dar, UrgGlobal, &globalArr);

  dispRg.clear();
}


double evalPhi(int cNum, int eType, int nodeNum,
    double psi, double eta, double gamma) {

  double phiVal = ot::ShapeFnCoeffs[cNum][eType][nodeNum][0] +
    (ot::ShapeFnCoeffs[cNum][eType][nodeNum][1]*psi) +
    (ot::ShapeFnCoeffs[cNum][eType][nodeNum][2]*eta) +
    (ot::ShapeFnCoeffs[cNum][eType][nodeNum][3]*gamma) +
    (ot::ShapeFnCoeffs[cNum][eType][nodeNum][4]*psi*eta) +
    (ot::ShapeFnCoeffs[cNum][eType][nodeNum][5]*eta*gamma) +
    (ot::ShapeFnCoeffs[cNum][eType][nodeNum][6]*gamma*psi) +
    (ot::ShapeFnCoeffs[cNum][eType][nodeNum][7]*psi*eta*gamma);

  return phiVal;
}

void enforceBC(ot::DA* da, unsigned char* bdyArr, Vec U) {
  if(da->iAmActive()) {
    PetscScalar* uArr;
    da->vecGetBuffer(U, uArr, false, false, false, 3);

    for(da->init<ot::DA_FLAGS::ALL>(); 
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>()) {
      unsigned int indices[8];
      da->getNodeIndices(indices);
      for(int k = 0; k < 8; k++) {
        if(bdyArr[indices[k]]) {
          for(int dof = 0; dof < 3; dof++) {
            uArr[(3*indices[k]) + dof] = 0.0;
          }//end if dof
        }//end if bdy 
      }//end for k
    }//end ALL loop

    da->vecRestoreBuffer(U, uArr, false, false, false, 3);
  }//end if active
}

void computeInvBlockDiagEntriesForHessMat(Mat J, double **invBlockDiagEntries) {
  ot::DAMG damg;
  MatShellGetContext(J, (void**)(&damg));
  assert(damg != NULL);

  HessData* data = (static_cast<HessData*>(damg->user));

  assert(data != NULL);

  ot::DA* da = damg->da;
  assert(da != NULL);

  unsigned int dof = 3;
  unsigned int nodeSize = damg->da->getNodeSize();

  //Initialize
  for(int i = 0; i < (dof*nodeSize); i++ ) {
    for(int j = 0; j < dof; j++) {
      invBlockDiagEntries[i][j] = 0.0;
    }//end for j
  }//end for i

  unsigned char* bdyArr = data->bdyArr;
  double**** LaplacianStencil = data->LaplacianStencil;
  double**** GradDivStencil = data->GradDivStencil;
  double mu = data->mu;
  double lambda = data->lambda;
  double alpha = data->alpha;
  unsigned int maxD;
  double hFac;

  int nineToSixMap[] = {0, 1, 2, 1, 3, 4, 2, 4, 5};

  std::vector<double> blockDiagVec;
  da->createVector<double>(blockDiagVec, false, false, 9);
  for(unsigned int i = 0; i < blockDiagVec.size(); i++) {
    blockDiagVec[i] = 0.0;
  }

  if(da->iAmActive()) {
    double *blockDiagArr;
    /*Nodal,Non-Ghosted,Write,9 dof*/
    da->vecGetBuffer<double>(blockDiagVec, blockDiagArr, false, false, false, 9);

    double *gtArr;
    assert( (data->gtVec) != NULL );
    /*Nodal,Non-Ghosted,Read,6 dof*/
    da->vecGetBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);

    maxD = (da->getMaxDepth());
    hFac = 1.0/((double)(1u << (maxD-1)));
    /*Loop through All Elements including ghosted*/
    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>()) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      double facImg = (h*h*h)/8.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0; k < 8; k++) {
          if(bdyArr[indices[k]]) {
            /*Dirichlet Node*/
            for(int dof = 0; dof < 3; dof++) {
              blockDiagArr[(9*indices[k]) + (3*dof) + dof] = alpha;
            } /*end dof*/
          } else { 
            /*Elasticity Part*/
            for(int dof = 0; dof < 3; dof++) {
              blockDiagArr[(9*indices[k])+(3*dof) + dof] += (alpha*mu*facElas*
                  LaplacianStencil[childNum][elemType][k][k]);
            } /*end dof*/
            for(int dofOut = 0; dofOut < 3; dofOut++) {
              for(int dofIn = 0; dofIn < 3; dofIn++) {
                blockDiagArr[(9*indices[k]) + (3*dofOut) + dofIn] +=
                  (alpha*(mu + lambda)*facElas*
                   GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*k) + dofIn]);
              } /*end dofIn*/
            } /*end dofOut*/
            /*Image Part*/
            for(int i = 0; i < 9; i++) {
              blockDiagArr[(9*indices[k]) + i] += 
                (facImg*gtArr[(6*indices[k]) + nineToSixMap[i]]);
            }
          } /*end if boundary*/
        } /*end k*/
    } /*end ALL*/

    da->vecRestoreBuffer<double>(blockDiagVec, blockDiagArr, false, false, false, 9);
    da->vecRestoreBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);
  } /*end if active*/

  for(unsigned int i = 0; i < nodeSize; i++) {
    double a11 = blockDiagVec[(9*i)];
    double a12 = blockDiagVec[(9*i)+1];
    double a13 = blockDiagVec[(9*i)+2];
    double a21 = blockDiagVec[(9*i)+3];
    double a22 = blockDiagVec[(9*i)+4];
    double a23 = blockDiagVec[(9*i)+5];
    double a31 = blockDiagVec[(9*i)+6];
    double a32 = blockDiagVec[(9*i)+7];
    double a33 = blockDiagVec[(9*i)+8];

    double detA = ((a11*a22*a33)-(a11*a23*a32)-(a21*a12*a33)
        +(a21*a13*a32)+(a31*a12*a23)-(a31*a13*a22));

    assert(fabs(detA) > 1e-12); 

    invBlockDiagEntries[3*i][0] = (a22*a33-a23*a32)/detA;

    invBlockDiagEntries[3*i][1] = -(a12*a33-a13*a32)/detA;

    invBlockDiagEntries[3*i][2] = (a12*a23-a13*a22)/detA;

    invBlockDiagEntries[(3*i)+1][0] = -(a21*a33-a23*a31)/detA;

    invBlockDiagEntries[(3*i)+1][1] = (a11*a33-a13*a31)/detA;

    invBlockDiagEntries[(3*i)+1][2] = -(a11*a23-a13*a21)/detA;

    invBlockDiagEntries[(3*i)+2][0] = (a21*a32-a22*a31)/detA;

    invBlockDiagEntries[(3*i)+2][1] = -(a11*a32-a12*a31)/detA;

    invBlockDiagEntries[(3*i)+2][2] = (a11*a22-a12*a21)/detA;
  }//end for i

  blockDiagVec.clear();
}


void getDofAndNodeSizeForHessMat(Mat J, unsigned int & dof, unsigned int & nodeSize) {
  ot::DAMG damg;
  MatShellGetContext(J, (void**)(&damg));
  dof = 3;
  nodeSize = damg->da->getNodeSize();
}

void getActiveStateAndActiveCommForKSP_Shell_Hess(Mat mat,
    bool & activeState, MPI_Comm & activeComm) {
  PetscTruth isshell;
  PetscTypeCompare((PetscObject)mat, MATSHELL, &isshell);
  assert(isshell);
  ot::DAMG damg;
  MatShellGetContext(mat, (void**)(&damg));
  assert(damg != NULL);
  ot::DA* da = damg->da;
  assert(da != NULL);
  activeState = da->iAmActive();
  activeComm = da->getCommActive();
}

void getPrivateMatricesForKSP_Shell_Hess(Mat mat,
    Mat *AmatPrivate, Mat *PmatPrivate, MatStructure* pFlag) {
  PetscTruth isshell;
  PetscTypeCompare((PetscObject)mat, MATSHELL, &isshell);
  assert(isshell);
  ot::DAMG damg;
  MatShellGetContext(mat, (void**)(&damg));
  assert(damg != NULL);
  HessData* data = (static_cast<HessData*>(damg->user));
  assert(data != NULL);
  *AmatPrivate = data->Jmat_private;
  *PmatPrivate = data->Jmat_private;
  *pFlag = DIFFERENT_NONZERO_PATTERN;
}

void detJacMaxAndMin(DA da,  Vec u, double* maxDetJac,  double* minDetJac) {
  PetscInt Ne;
  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  PetscScalar**** uLocalArr;

  //DA returns the number of nodes.
  //Need the number of elements.
  DAGetInfo(da, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL);
  Ne--;

  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  PetscInt nxe, nye, nze;

  if((xs + nx) == (Ne + 1)) {
    nxe = nx - 1;
  } else {
    nxe = nx;
  }

  if((ys + ny) == (Ne + 1)) {
    nye = ny - 1;
  } else {
    nye = ny;
  }

  if((zs + nz) == (Ne + 1)) {
    nze = nz - 1;
  } else {
    nze = nz;
  }

  double h = 1.0/static_cast<double>(Ne);

  Vec uLocal;
  DAGetLocalVector(da, &uLocal);

  DAGlobalToLocalBegin(da, u, INSERT_VALUES, uLocal);
  DAGlobalToLocalEnd(da, u, INSERT_VALUES, uLocal);

  DAVecGetArrayDOF(da, uLocal, &uLocalArr);

  double Minv[][8] = { { 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250},
    { -0.1250, 0.1250, -0.1250, 0.1250, -0.1250, 0.1250, -0.1250, 0.1250},
    { -0.1250, -0.1250, 0.1250, 0.1250, -0.1250, -0.1250, 0.1250, 0.1250},
    { -0.1250, -0.1250, -0.1250, -0.1250, 0.1250, 0.1250, 0.1250, 0.1250},
    { 0.1250, -0.1250, -0.1250, 0.1250, 0.1250, -0.1250, -0.1250, 0.1250},
    { 0.1250, 0.1250, -0.1250, -0.1250, -0.1250, -0.1250, 0.1250, 0.1250},
    { 0.1250, -0.1250, 0.1250, -0.1250, -0.1250, 0.1250, -0.1250, 0.1250},
    { -0.1250, 0.1250, 0.1250, -0.1250, 0.1250, -0.1250, -0.1250, 0.1250} };

  double minDetJacLocal;
  double maxDetJacLocal;

  int numEvalPts = 7;
  double psiArr[] = {0, 0.5, -0.5, 0, 0, 0, 0};
  double etaArr[] = {0, 0, 0, 0.5, -0.5, 0, 0};
  double gammaArr[] = {0, 0, 0, 0, 0, 0.5, -0.5};

  {
    //First Element
    int ilist[8];
    int jlist[8];
    int klist[8];
    double coeffs[3][8];
    double vertices[3][8];
    for(int r = 0; r < 8; r++) {
      ilist[r] = xs + (r%2);
      jlist[r] = ys + ((r/2)%2);
      klist[r] = zs + (r/4);
    }
    for(int r = 0; r < 8; r++) {
      double xyzPos[3];
      xyzPos[0] = static_cast<double>(ilist[r])*h;
      xyzPos[1] = static_cast<double>(jlist[r])*h;
      xyzPos[2] = static_cast<double>(klist[r])*h;
      for(int d = 0; d < 3; d++) {
        vertices[d][r] = xyzPos[d] + uLocalArr[klist[r]][jlist[r]][ilist[r]][d];
      }
    }
    for(int d = 0; d < 3; d++) {
      for(int r = 0; r < 8; r++) {
        coeffs[d][r] = 0;
      }
    }
    for(int d = 0; d < 3; d++) {
      for(int r = 0; r < 8; r++) {
        for(int c = 0; c < 8; c++) {
          coeffs[d][r] += (Minv[r][c]*vertices[d][c]);
        }
      }
    }
    double psi = 0;
    double eta = 0;
    double gamma = 0;
    double Jmat[3][3];
    for(int d = 0; d < 3; d++) {
      Jmat[d][0] = DiffX(coeffs[d], psi, eta, gamma); 
      Jmat[d][1] = DiffY(coeffs[d], psi, eta, gamma); 
      Jmat[d][2] = DiffZ(coeffs[d], psi, eta, gamma); 
    }
    double detJacLocal = ( ( (Jmat[0][0]*Jmat[1][1]*Jmat[2][2]) +
          (Jmat[0][1]*Jmat[1][2]*Jmat[2][0]) + (Jmat[0][2]*Jmat[1][0]*Jmat[2][1]) ) -
        ( (Jmat[2][0]*Jmat[1][1]*Jmat[0][2]) + (Jmat[2][1]*Jmat[1][2]*Jmat[0][0]) +
          (Jmat[2][2]*Jmat[1][0]*Jmat[0][1]) ) );
    minDetJacLocal = detJacLocal;
    maxDetJacLocal = detJacLocal;
  }

  for(int k = zs; k < zs + nze; k++) {
    for(int j = ys; j < ys + nye; j++) {
      for(int i = xs; i < xs + nxe; i++) {
        double coeffs[3][8];
        double vertices[3][8];
        int ilist[8];
        int jlist[8];
        int klist[8];
        for(int r = 0; r < 8; r++) {
          ilist[r] = i + (r%2);
          jlist[r] = j + ((r/2)%2);
          klist[r] = k + (r/4);
        }
        for(int r = 0; r < 8; r++) {
          double xyzPos[3];
          xyzPos[0] = static_cast<double>(ilist[r])*h;
          xyzPos[1] = static_cast<double>(jlist[r])*h;
          xyzPos[2] = static_cast<double>(klist[r])*h;
          for(int d = 0; d < 3; d++) {
            vertices[d][r] = xyzPos[d] + uLocalArr[klist[r]][jlist[r]][ilist[r]][d];
          }
        }
        for(int d = 0; d < 3; d++) {
          for(int r = 0; r < 8; r++) {
            coeffs[d][r] = 0;
          }
        }
        for(int d = 0; d < 3; d++) {
          for(int r = 0; r < 8; r++) {
            for(int c = 0; c < 8; c++) {
              coeffs[d][r] += (Minv[r][c]*vertices[d][c]);
            }
          }
        } 
        for(int n = 0; n < numEvalPts; n++) {
          double psi = psiArr[n];
          double eta = etaArr[n];
          double gamma = gammaArr[n];
          double Jmat[3][3]; 
          for(int d = 0; d < 3; d++) {
            Jmat[d][0] = DiffX(coeffs[d], psi, eta, gamma); 
            Jmat[d][1] = DiffY(coeffs[d], psi, eta, gamma); 
            Jmat[d][2] = DiffZ(coeffs[d], psi, eta, gamma); 
          } 
          double detJacLocal = ( ( (Jmat[0][0]*Jmat[1][1]*Jmat[2][2]) +
                (Jmat[0][1]*Jmat[1][2]*Jmat[2][0]) + (Jmat[0][2]*Jmat[1][0]*Jmat[2][1]) ) -
              ( (Jmat[2][0]*Jmat[1][1]*Jmat[0][2]) + (Jmat[2][1]*Jmat[1][2]*Jmat[0][0]) +
                (Jmat[2][2]*Jmat[1][0]*Jmat[0][1]) ) );
          if(detJacLocal < minDetJacLocal) {
            minDetJacLocal = detJacLocal;
          }
          if(detJacLocal > maxDetJacLocal) {
            maxDetJacLocal = detJacLocal;
          }
        }
      }
    }
  }

  DAVecRestoreArrayDOF(da, uLocal, &uLocalArr);

  DARestoreLocalVector(da, &uLocal);

  MPI_Comm comm;
  PetscObjectGetComm((PetscObject) da, &comm);

  minDetJacLocal *= 8.0/(h*h*h);
  maxDetJacLocal *= 8.0/(h*h*h);

  par::Mpi_Allreduce<double>(&minDetJacLocal, minDetJac, 1, MPI_MIN, comm);
  par::Mpi_Allreduce<double>(&maxDetJacLocal, maxDetJac, 1, MPI_MAX, comm);

}

void computeFDgradient(DA dai, DA dao, Vec in, Vec out) {
  PetscInt Ne;
  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  PetscScalar**** iArr;
  PetscScalar**** oArr;
  Vec iLoc;
  int dof; //Number of components per node

  DAGetLocalVector(dai, &iLoc);
  DAGlobalToLocalBegin(dai, in, INSERT_VALUES, iLoc);

  //DA returns the number of nodes.
  //Need the number of elements.
  DAGetInfo(dai, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, &dof, PETSC_NULL,
      PETSC_NULL, PETSC_NULL);
  Ne--;

  DAGetCorners(dai, &xs, &ys, &zs, &nx, &ny, &nz);

  double h = 1.0/static_cast<double>(Ne);

  VecZeroEntries(out);
  DAVecGetArrayDOF(dao, out, &oArr);

  DAGlobalToLocalEnd(dai, in, INSERT_VALUES, iLoc);

  DAVecGetArrayDOF(dai, iLoc, &iArr);
  for(int k = zs; k < zs + nz; k++) {
    for(int j = ys; j < ys + ny; j++) {
      for(int i = xs; i < xs + nx; i++) {
        for(int d = 0; d < dof; d++) {
          PetscScalar fxplus = 0;
          PetscScalar fyplus = 0;
          PetscScalar fzplus = 0;
          PetscScalar fxminus = 0;
          PetscScalar fyminus = 0;
          PetscScalar fzminus = 0;
          //Only need to take care of domain boundaries
          //Processor boundaries are okay, since we work with ghosted (local) vectors
          if(i) {
            fxminus = iArr[k][j][i - 1][d];
          }
          if(i < Ne) {
            fxplus = iArr[k][j][i + 1][d];
          }
          if(j) {
            fyminus = iArr[k][j - 1][i][d];
          }
          if(j < Ne) {
            fyplus = iArr[k][j + 1][i][d];
          }
          if(k) {
            fzminus = iArr[k - 1][j][i][d];
          }
          if(k < Ne) {
            fzplus = iArr[k + 1][j][i][d];
          }
          oArr[k][j][i][(3*d) + 0] = (fxplus - fxminus)/(2.0*h);
          oArr[k][j][i][(3*d) + 1] = (fyplus - fyminus)/(2.0*h);
          oArr[k][j][i][(3*d) + 2] = (fzplus - fzminus)/(2.0*h);
        }//end for d
      }//end for i
    }//end for j
  }//end for k

  DAVecRestoreArrayDOF(dai, iLoc, &iArr);
  DARestoreLocalVector(dai, &iLoc);
  DAVecRestoreArrayDOF(dao, out, &oArr);
}

void computeFDhessian(DA dai, DA dao, Vec in, Vec out) {
  PetscInt Ne;
  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  PetscScalar**** iArr;
  PetscScalar**** oArr;
  Vec iLoc;
  int dof; //Number of components per node

  DAGetLocalVector(dai, &iLoc);

  DAGlobalToLocalBegin(dai, in, INSERT_VALUES, iLoc);

  //DA returns the number of nodes.
  //Need the number of elements.
  DAGetInfo(dai, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, &dof, PETSC_NULL,
      PETSC_NULL, PETSC_NULL);
  Ne--;

  DAGetCorners(dai, &xs, &ys, &zs, &nx, &ny, &nz);

  double h = 1.0/static_cast<double>(Ne);
  double hsq = h*h;

  VecZeroEntries(out);

  DAVecGetArrayDOF(dao, out, &oArr);

  DAGlobalToLocalEnd(dai, in, INSERT_VALUES, iLoc);

  DAVecGetArrayDOF(dai, iLoc, &iArr);
  for(int k = zs; k < zs + nz; k++) {
    for(int j = ys; j < ys + ny; j++) {
      for(int i = xs; i < xs + nx; i++) {
        for(int d = 0; d < dof; d++) {
          PetscScalar fxplus = 0;
          PetscScalar fyplus = 0;
          PetscScalar fzplus = 0;
          PetscScalar fxminus = 0;
          PetscScalar fyminus = 0;
          PetscScalar fzminus = 0;
          PetscScalar fxpyp = 0;
          PetscScalar fxpym = 0;
          PetscScalar fxpzp = 0;
          PetscScalar fxpzm = 0;
          PetscScalar fxmyp = 0;
          PetscScalar fxmym = 0;
          PetscScalar fxmzp = 0;
          PetscScalar fxmzm = 0;
          PetscScalar fypzp = 0;
          PetscScalar fypzm = 0;
          PetscScalar fymzp = 0;
          PetscScalar fymzm = 0;
          //Only need to take care of domain boundaries
          //Processor boundaries are okay, since we work with ghosted (local) vectors
          if(i) {
            fxminus = iArr[k][j][i - 1][d];
            if(j < Ne) {
              fxmyp = iArr[k][j + 1][i - 1][d];
            }
            if(j) {
              fxmym = iArr[k][j - 1][i - 1][d];
            }
            if(k < Ne) {
              fxmzp = iArr[k + 1][j][i - 1][d];
            }
            if(k) {
              fxmzm = iArr[k - 1][j][i - 1][d];
            }
          }
          if(i < Ne) {
            fxplus = iArr[k][j][i + 1][d];
            if(j < Ne) {
              fxpyp = iArr[k][j + 1][i + 1][d];
            }
            if (j) {
              fxpym = iArr[k][j - 1][i + 1][d];
            }
            if (k < Ne) {
              fxpzp = iArr[k + 1][j][i + 1][d];
            }
            if (k) {
              fxpzm = iArr[k - 1][j][i + 1][d];
            }
          }
          if(j) {
            fyminus = iArr[k][j - 1][i][d];
            if(k < Ne) {
              fymzp = iArr[k + 1][j - 1][i][d];
            }
            if(k) {
              fymzm = iArr[k - 1][j - 1][i][d];
            }
          }
          if(j < Ne) {
            fyplus = iArr[k][j + 1][i][d];
            if(k < Ne) {
              fypzp = iArr[k + 1][j + 1][i][d];
            }
            if(k) {
              fypzm = iArr[k - 1][j + 1][i][d];
            } 
          }
          if(k) {
            fzminus = iArr[k - 1][j][i][d];
          }
          if(k < Ne) {
            fzplus = iArr[k + 1][j][i][d];
          }
          //h12
          oArr[k][j][i][(6*d) + 0] = (fxpyp + fxmym - (fxpym + fxmyp))/(4.0*hsq);

          //h13
          oArr[k][j][i][(6*d) + 1] =  (fxpzp + fxmzm - (fxpzm + fxmzp))/(4.0*hsq);

          //h23
          oArr[k][j][i][(6*d) + 2] =  (fypzp + fymzm - (fypzm + fymzp))/(4.0*hsq);

          //h11
          oArr[k][j][i][(6*d) + 3] =  (fxplus + fxminus - (2.0*iArr[k][j][i][d]))/hsq;

          //h22
          oArr[k][j][i][(6*d) + 4] =  (fyplus + fyminus - (2.0*iArr[k][j][i][d]))/hsq;

          //h33
          oArr[k][j][i][(6*d) + 5] =  (fzplus + fzminus - (2.0*iArr[k][j][i][d]))/hsq;
        }//end for d
      }//end for i
    }//end for j
  }//end for k

  DAVecRestoreArrayDOF(dai, iLoc, &iArr);
  DARestoreLocalVector(dai, &iLoc);
  DAVecRestoreArrayDOF(dao, out, &oArr);
}

bool foundValidDApart(int N, int npes) {
  int m, n, p;

  n = (int)(0.5 + pow(static_cast<double>(npes),(1.0/3.0)));

  if(!n) { 
    n = 1;
  }

  while (n > 0) {
    int pm = npes/n;

    if (n*pm == npes) {
      break;
    }

    n--;
  }

  if (!n) {
    n = 1; 
  }

  m = (int)(0.5 + sqrt(static_cast<double>(npes)/static_cast<double>(n)));

  if(!m) { 
    m = 1; 
  }

  while (m > 0) {
    p = npes/(m*n);
    if (m*n*p == npes) {
      break;
    }
    m--;
  }

  bool foundValidPart = true;
  if((m*n*p) != npes) {
    foundValidPart = false;
  }
  if(N < m) {
    foundValidPart = false;
  }
  if(N < n) {
    foundValidPart = false;
  }
  if(N < p) {
    foundValidPart = false;
  }

  return foundValidPart;
}

void createImagePatches(DA dar, ot::DA* dao, int padding, int numImages,
    const std::vector<double >& sigGlobal, 
    const std::vector<double >& tauGlobal,
    const std::vector<double >& gradSigGlobal,
    const std::vector<double >& gradTauGlobal,
    std::vector<std::vector<double> >& sigLocal,
    std::vector<std::vector<double> >& tauLocal,
    std::vector<std::vector<double> >& gradSigLocal,
    std::vector<std::vector<double> >& gradTauLocal) {

  PetscLogEventBegin(createPatchesEvent, 0, 0, 0, 0);

  //Active processors for the Octree mesh must be a subset or equal to that for
  //the DA mesh

  MPI_Comm comm;
  PetscObjectGetComm((PetscObject)dar, &comm);

  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  PetscInt npx, npy, npz;

  //Do Not Free lx, ly, lz. They are managed by DA
  PetscInt Ne;

  const PetscInt* lx = NULL;
  const PetscInt* ly = NULL;
  const PetscInt* lz = NULL;

  //Number of nodes is 1 more than the number of elements
  DAGetOwnershipRanges(dar, &lx, &ly, &lz);
  DAGetInfo(dar, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL, &npx, &npy, &npz,	
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  Ne--; 

  assert(lx != NULL);
  assert(ly != NULL);
  assert(lz != NULL);

  double h = 1.0/static_cast<double>(Ne);

  std::vector<double> scanLx(npx);
  std::vector<double> scanLy(npy);
  std::vector<double> scanLz(npz);

  scanLx[0] = 0;
  scanLy[0] = 0;
  scanLz[0] = 0;
  for(int i = 1; i < npx; i++) {
    scanLx[i] = scanLx[i - 1] + (static_cast<double>(lx[i - 1])*h);
  }
  for(int i = 1; i < npy; i++) {
    scanLy[i] = scanLy[i - 1] + (static_cast<double>(ly[i - 1])*h);
  }
  for(int i = 1; i < npz; i++) {
    scanLz[i] = scanLz[i - 1] + (static_cast<double>(lz[i - 1])*h);
  }

  std::vector<ot::TreeNode> blocks;
  unsigned int maxDepth;
  double hOctFac;

  if((dao != NULL) && (dao->iAmActive())) {
    blocks = dao->getBlocks();
    maxDepth = dao->getMaxDepth();
    hOctFac = 1.0/(static_cast<double>(1u<<(maxDepth - 1)));
  }

  int numBlocks = blocks.size();

  std::vector<double>* tmpSendPts = new std::vector<double>[npes];
  assert(tmpSendPts != NULL);

  std::vector<int>* ptMap = new std::vector<int>[npes];
  assert(ptMap != NULL);

  std::vector<int> blockPtCnt(numBlocks);

  unsigned int numSendPts = 0;
  for(int i = 0; i < numBlocks; i++) {
    unsigned int minX = blocks[i].getX();
    unsigned int minY = blocks[i].getY();
    unsigned int minZ = blocks[i].getZ();

    unsigned int maxX = blocks[i].maxX();
    unsigned int maxY = blocks[i].maxY();
    unsigned int maxZ = blocks[i].maxZ();

    double xsPt = hOctFac*(static_cast<double>(minX));
    double ysPt = hOctFac*(static_cast<double>(minY));
    double zsPt = hOctFac*(static_cast<double>(minZ));

    double xePt = hOctFac*(static_cast<double>(maxX));
    double yePt = hOctFac*(static_cast<double>(maxY));
    double zePt = hOctFac*(static_cast<double>(maxZ));

    double xs = (xsPt - ((static_cast<double>(padding))*h));
    double ys = (ysPt - ((static_cast<double>(padding))*h));
    double zs = (zsPt - ((static_cast<double>(padding))*h));

    double xe = (xePt + ((static_cast<double>(padding))*h));
    double ye = (yePt + ((static_cast<double>(padding))*h));
    double ze = (zePt + ((static_cast<double>(padding))*h));

    if( xs < 0.0 ) {
      xs = 0.0;
    }
    if( ys < 0.0 ) {
      ys = 0.0;
    }
    if( zs < 0.0 ) {
      zs = 0.0;
    }

    if(xe > 1.0) {
      xe = 1.0;
    }
    if(ye > 1.0) {
      ye = 1.0;
    }
    if(ze > 1.0) {
      ze = 1.0;
    }

    int pCnt = 0;
    for(double zc = zs; zc <= ze; zc += h) {
      for(double yc = ys; yc <= ye; yc += h) {
        for(double xc = xs; xc <= xe; xc += h) {
          unsigned int xRes, yRes, zRes;
          seq::maxLowerBound<double>(scanLx, xc, xRes, 0, 0);
          seq::maxLowerBound<double>(scanLy, yc, yRes, 0, 0);
          seq::maxLowerBound<double>(scanLz, zc, zRes, 0, 0);
          unsigned int toSendRank = (((zRes*npy) + yRes)*npx) + xRes;
          assert(toSendRank < npes);
          tmpSendPts[toSendRank].push_back(xc);
          tmpSendPts[toSendRank].push_back(yc);
          tmpSendPts[toSendRank].push_back(zc);
          ptMap[toSendRank].push_back(i);
          ptMap[toSendRank].push_back(pCnt);
          pCnt++;
        }
      }
    }

    blockPtCnt[i] = pCnt;

    numSendPts += pCnt;
  }

  std::vector<double> sendPts;
  sendPts.reserve(3*numSendPts);

  int* sendCnts = new int[npes];
  int* sendOff = new int[npes];
  for(int i = 0; i < npes; i++) {
    sendCnts[i] = tmpSendPts[i].size();
    sendOff[i] = sendPts.size();
    if(sendCnts[i]) {
      sendPts.insert(sendPts.end(), tmpSendPts[i].begin(), tmpSendPts[i].end());
    }
  }

  delete [] tmpSendPts;
  tmpSendPts = NULL;

  int* recvCnts = new int[npes];

  par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, comm);

  int* recvOff = new int[npes];
  recvOff[0] = 0;
  for(int i = 1; i < npes; i++) {
    recvOff[i] = recvOff[i - 1] + recvCnts[i - 1];
  }

  std::vector<double> recvPts(recvOff[npes - 1] + recvCnts[npes - 1]);

  //Dense would probably be better here...
  par::Mpi_Alltoallv_sparse<double>((&(*(sendPts.begin()))), sendCnts, sendOff,
      (&(*(recvPts.begin()))), recvCnts, recvOff, comm);

  PetscInt nx, ny, nz;
  PetscInt xi, yi, zi;
  DAGetCorners(dar, &xi, &yi, &zi, &nx, &ny, &nz);

  unsigned int numRecvPts = (recvPts.size())/3;

  std::vector<double> sigRes(numRecvPts*numImages);
  std::vector<double> tauRes(numRecvPts*numImages);
  std::vector<double> gradSigRes(3*numRecvPts*numImages);
  std::vector<double> gradTauRes(3*numRecvPts*numImages);

  for(int i = 0; i < numRecvPts; i++) {
    double xPt = recvPts[3*i];
    double yPt = recvPts[(3*i) + 1];
    double zPt = recvPts[(3*i) + 2];

    int xid = static_cast<int>(xPt/h) - xi;
    int yid = static_cast<int>(yPt/h) - yi;
    int zid = static_cast<int>(zPt/h) - zi;

    assert(xid >= 0);
    assert(yid >= 0);
    assert(zid >= 0);

    assert(xid < nx);
    assert(yid < ny);
    assert(zid < nz);

    int globArrIdx = ((((zid*ny) + yid)*nx) + xid);

    assert( (numImages*globArrIdx) < sigGlobal.size() );
    assert( (numImages*globArrIdx) < tauGlobal.size() );
    assert( (3*numImages*globArrIdx) < gradSigGlobal.size() );
    assert( (3*numImages*globArrIdx) < gradTauGlobal.size() );

    for(int j = 0; j < numImages; j++) {
      sigRes[(numImages*i) + j] = sigGlobal[(numImages*globArrIdx) + j];
      tauRes[(numImages*i) + j] = tauGlobal[(numImages*globArrIdx) + j];
      for(int d = 0; d < 3; d++) {
        gradSigRes[(3*((numImages*i) + j)) + d] = gradSigGlobal[(3*((numImages*globArrIdx) + j)) + d];
        gradTauRes[(3*((numImages*i) + j)) + d] = gradTauGlobal[(3*((numImages*globArrIdx) + j)) + d];
      }
    }
  }

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = ((sendCnts[i])*numImages);
    sendOff[i] = ((sendOff[i])*numImages);
    recvCnts[i] = ((recvCnts[i])*numImages);
    recvOff[i] = ((recvOff[i])*numImages);
  }

  std::vector<double> gradSigTmp(3*numSendPts*numImages);
  std::vector<double> gradTauTmp(3*numSendPts*numImages);

  //Dense would probably be better here...
  par::Mpi_Alltoallv_sparse<double>( (&(*(gradSigRes.begin()))), recvCnts, recvOff,
      (&(*(gradSigTmp.begin()))), sendCnts, sendOff, comm);

  par::Mpi_Alltoallv_sparse<double>( (&(*(gradTauRes.begin()))), recvCnts, recvOff,
      (&(*(gradTauTmp.begin()))), sendCnts, sendOff, comm);

  gradSigRes.clear();
  gradTauRes.clear();

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = ((sendCnts[i])/3);
    sendOff[i] = ((sendOff[i])/3);
    recvCnts[i] = ((recvCnts[i])/3);
    recvOff[i] = ((recvOff[i])/3);
  }

  std::vector<double> sigTmp(numSendPts*numImages);
  std::vector<double> tauTmp(numSendPts*numImages);

  //Dense would probably be better here...
  par::Mpi_Alltoallv_sparse<double>( (&(*(sigRes.begin()))), recvCnts, recvOff,
      (&(*(sigTmp.begin()))), sendCnts, sendOff, comm);

  par::Mpi_Alltoallv_sparse<double>( (&(*(tauRes.begin()))), recvCnts, recvOff,
      (&(*(tauTmp.begin()))), sendCnts, sendOff, comm);

  delete [] recvCnts;
  recvCnts = NULL;

  delete [] recvOff;
  recvOff = NULL;

  sigRes.clear();
  tauRes.clear();

  sigLocal.resize(numBlocks);
  tauLocal.resize(numBlocks);
  gradSigLocal.resize(numBlocks);
  gradTauLocal.resize(numBlocks);

  for(int i = 0; i < numBlocks; i++) {
    sigLocal[i].resize(numImages*(blockPtCnt[i]));
    tauLocal[i].resize(numImages*(blockPtCnt[i]));
    gradSigLocal[i].resize(3*numImages*(blockPtCnt[i]));
    gradTauLocal[i].resize(3*numImages*(blockPtCnt[i]));
  }

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = ((sendCnts[i])/numImages);
    sendOff[i] = ((sendOff[i])/numImages);
  }

  for(int i = 0; i < npes; i++) {
    for(int j = 0; j < sendCnts[i]; j++) {
      int blockId = ptMap[i][2*j];
      int ptIdxInBlock = ptMap[i][(2*j) + 1];
      for(int k = 0; k < numImages; k++) {
        sigLocal[blockId][(numImages*ptIdxInBlock) + k] = sigTmp[(numImages*(sendOff[i] + j)) + k];
        tauLocal[blockId][(numImages*ptIdxInBlock) + k] = tauTmp[(numImages*(sendOff[i] + j)) + k];
        for(int d = 0; d < 3; d++) {
          gradSigLocal[blockId][(3*((numImages*ptIdxInBlock) + k)) + d] = gradSigTmp[(3*((numImages*(sendOff[i] + j)) + k)) + d];
          gradTauLocal[blockId][(3*((numImages*ptIdxInBlock) + k)) + d] = gradTauTmp[(3*((numImages*(sendOff[i] + j)) + k)) + d];
        }
      }
    }
  }

  sigTmp.clear();
  tauTmp.clear();
  gradSigTmp.clear();
  gradTauTmp.clear();

  delete [] ptMap;
  ptMap = NULL;

  delete [] sendCnts;
  sendCnts = NULL;

  delete [] sendOff;
  sendOff = NULL;

  PetscLogEventEnd(createPatchesEvent, 0, 0, 0, 0);

}//end fn.


void zeroDouble(double & v) {
  v = 0;
}


int createPhimat(double******& Phimat, int numGpts, double* gPts) {

  typedef double* doublePtr;
  typedef doublePtr* double2Ptr;
  typedef double2Ptr* double3Ptr;
  typedef double3Ptr* double4Ptr;
  typedef double4Ptr* double5Ptr;

  Phimat = new double5Ptr[8];
  for(int cNum = 0; cNum < 8; cNum++) {
    Phimat[cNum] = new double4Ptr[18];
    for(int eType = 0; eType < 18; eType++) {
      Phimat[cNum][eType] = new double3Ptr[8];
      for(int i = 0; i < 8; i++) {
        Phimat[cNum][eType][i] = new double2Ptr[numGpts];
        for(int m = 0; m < numGpts; m++) {
          Phimat[cNum][eType][i][m] = new doublePtr[numGpts];
          for(int n = 0; n < numGpts; n++) {
            Phimat[cNum][eType][i][m][n] = new double[numGpts];
            for(int p = 0; p < numGpts; p++) {
              Phimat[cNum][eType][i][m][n][p] = evalPhi(cNum, eType, i, gPts[m], gPts[n], gPts[p]);
            }
          }
        }
      }
    }
  }

  return 1;
}


int createGDmat(double****& GDmat) {

#ifdef __USE_MG_INIT_TYPE3__
  createGDmat_Type3(GDmat);
#else
#ifdef __USE_MG_INIT_TYPE2__
  createGDmat_Type2(GDmat);
#else
  createGDmat_Type1(GDmat);
#endif
#endif

  return 1;
}


int createGDmat_Type3(double****& GDmat) {
  FILE* infile;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char fname[250];
  sprintf(fname,"GDType2Stencils_%d.inp", rank);
  infile = fopen(fname,"r");
  if(!infile) {
    std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
    assert(false);
  }

  typedef double* doublePtr;
  typedef doublePtr* double2Ptr;
  typedef double2Ptr* double3Ptr;

  GDmat = new double3Ptr[8];
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    GDmat[cNum] = new double2Ptr[18];
    for(unsigned int eType = 0; eType < 18; eType++) {
      GDmat[cNum][eType] = new doublePtr[24];
      for(unsigned int i = 0; i < 24; i++) {
        GDmat[cNum][eType][i] = new double[24];
        for(unsigned int j = 0; j < 24; j++) {
          fscanf(infile,"%lf",&(GDmat[cNum][eType][i][j]));
        }
      }
    }
  }
  fclose(infile);
  return 1;
}


int createGDmat_Type2(double ****& GDmat) {
  FILE* infile;
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  const int THOUSAND = 1000;
  int numGroups = (npes/THOUSAND);
  if( (numGroups*THOUSAND) < npes) {
    numGroups++;
  }

  MPI_Comm newComm;

  bool* isEmptyList = new bool[npes];
  for(int i = 0; i < numGroups; i++) {
    for(int j = 0; (j < (i*THOUSAND)) && (j < npes); j++) {
      isEmptyList[j] = true;
    }
    for(int j = (i*THOUSAND); (j < ((i+1)*THOUSAND)) && (j < npes); j++) {
      isEmptyList[j] = false;
    }
    for(int j = ((i + 1)*THOUSAND); j < npes; j++) {
      isEmptyList[j] = true;
    }
    MPI_Comm tmpComm;
    par::splitComm2way(isEmptyList, &tmpComm, comm);
    if(!(isEmptyList[rank])) {
      newComm = tmpComm;
    }
  }//end for i
  delete [] isEmptyList;
  isEmptyList = NULL;


  if((rank % THOUSAND) == 0) {
    char fname[250];
    sprintf(fname,"GDType2Stencils_%d.inp", (rank/THOUSAND));
    infile = fopen(fname,"r");
    if(!infile) {
      std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
      assert(false);
    }
  }

  typedef double* doublePtr;
  typedef doublePtr* double2Ptr;
  typedef double2Ptr* double3Ptr;

  GDmat = new double3Ptr[8];
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    GDmat[cNum] = new double2Ptr[18];
    for(unsigned int eType = 0; eType < 18; eType++) {
      GDmat[cNum][eType] = new doublePtr[24];
      for(unsigned int i = 0; i < 24; i++) {
        GDmat[cNum][eType][i] = new double[24];
        if((rank % THOUSAND) == 0) {
          for(unsigned int j = 0; j < 24; j++) {
            fscanf(infile,"%lf",&(GDmat[cNum][eType][i][j]));
          }
        }
      }
    }
  }

  if((rank % THOUSAND) == 0) {
    fclose(infile);
  }

  double * tmpMat = new double[82944];

  if((rank % THOUSAND) == 0) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 24; i++) {
          for(unsigned int j = 0; j < 24; j++) {
            tmpMat[ctr] = GDmat[cNum][eType][i][j];
            ctr++;
          }
        }
      }
    }
  }

  par::Mpi_Bcast<double>(tmpMat,82944, 0, newComm);

  if((rank % THOUSAND) != 0) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 24; i++) {
          for(unsigned int j = 0; j < 24; j++) {
            GDmat[cNum][eType][i][j] = tmpMat[ctr];
            ctr++;
          }
        }
      }
    }
  }

  delete [] tmpMat;
  tmpMat = NULL;

  return 1;
}//end fn.


/*Type 2 Matrices: Coarse and Fine are the same.*/
int createGDmat_Type1(double ****& GDmat) {
  FILE* infile;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(!rank) {
    char fname[250];
    sprintf(fname,"GDType2Stencils.inp");
    infile = fopen(fname,"r");
    if(!infile) {
      std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
      assert(false);
    }
  }

  typedef double* doublePtr;
  typedef doublePtr* double2Ptr;
  typedef double2Ptr* double3Ptr;

  GDmat = new double3Ptr[8];
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    GDmat[cNum] = new double2Ptr[18];
    for(unsigned int eType = 0; eType < 18; eType++) {
      GDmat[cNum][eType] = new doublePtr[24];
      for(unsigned int i = 0; i < 24; i++) {
        GDmat[cNum][eType][i] = new double[24];
        if(!rank) {
          for(unsigned int j = 0; j < 24; j++) {
            fscanf(infile,"%lf",&(GDmat[cNum][eType][i][j]));
          }
        }
      }
    }
  }

  if(!rank) {
    fclose(infile);
  }

  double * tmpMat = new double[82944];

  if(!rank) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 24; i++) {
          for(unsigned int j = 0; j < 24; j++) {
            tmpMat[ctr] = GDmat[cNum][eType][i][j];
            ctr++;
          }
        }
      }
    }
  }

  par::Mpi_Bcast<double>(tmpMat,82944, 0, MPI_COMM_WORLD);

  if(rank) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 24; i++) {
          for(unsigned int j = 0; j < 24; j++) {
            GDmat[cNum][eType][i][j] = tmpMat[ctr];
            ctr++;
          }
        }
      }
    }
  }

  delete [] tmpMat;
  tmpMat = NULL;
  return 1;
}//end fn.


int createLmat(double ****& Lmat) {

#ifdef __USE_MG_INIT_TYPE3__
  createLmat_Type3(Lmat);
#else
#ifdef __USE_MG_INIT_TYPE2__
  createLmat_Type2(Lmat);
#else
  createLmat_Type1(Lmat);
#endif
#endif
  return 1;
}


int createLmat_Type3(double ****& Lmat) {
  FILE* infile;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char fname[250];
  sprintf(fname,"LapType2Stencils_%d.inp", rank);
  infile = fopen(fname,"r");
  if(!infile) {
    std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
    assert(false);
  }

  typedef double* doublePtr;
  typedef doublePtr* double2Ptr;
  typedef double2Ptr* double3Ptr;

  Lmat = new double3Ptr[8];
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    Lmat[cNum] = new double2Ptr[18];
    for(unsigned int eType = 0; eType < 18; eType++) {
      Lmat[cNum][eType] = new doublePtr[8];
      for(unsigned int i = 0; i < 8; i++) {
        Lmat[cNum][eType][i] = new double[8];
        for(unsigned int j = 0; j < 8; j++) {
          fscanf(infile,"%lf",&(Lmat[cNum][eType][i][j]));
        }
      }
    }
  }

  fclose(infile);
  return 1;
}


int createLmat_Type2(double ****& Lmat) {
  FILE* infile;
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  const int THOUSAND = 1000;
  int numGroups = (npes/THOUSAND);
  if( (numGroups*THOUSAND) < npes) {
    numGroups++;
  }

  MPI_Comm newComm;

  bool* isEmptyList = new bool[npes];
  assert(isEmptyList);
  for(int i = 0; i < numGroups; i++) {
    for(int j = 0; (j < (i*THOUSAND)) && (j < npes); j++) {
      isEmptyList[j] = true;
    }
    for(int j = (i*THOUSAND); (j < ((i+1)*THOUSAND)) && (j < npes); j++) {
      isEmptyList[j] = false;
    }
    for(int j = ((i + 1)*THOUSAND); j < npes; j++) {
      isEmptyList[j] = true;
    }
    MPI_Comm tmpComm;
    par::splitComm2way(isEmptyList, &tmpComm, comm);
    if(!(isEmptyList[rank])) {
      newComm = tmpComm;
    }
  }//end for i
  delete [] isEmptyList;
  isEmptyList = NULL;


  if((rank % THOUSAND) == 0) {
    char fname[250];
    sprintf(fname,"LapType2Stencils_%d.inp", (rank/THOUSAND));
    infile = fopen(fname,"r");
    if(!infile) {
      std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
      assert(false);
    }
  }

  typedef double* doublePtr;
  typedef doublePtr* double2Ptr;
  typedef double2Ptr* double3Ptr;

  Lmat = new double3Ptr[8];
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    Lmat[cNum] = new double2Ptr[18];
    for(unsigned int eType = 0; eType < 18; eType++) {
      Lmat[cNum][eType] = new doublePtr[8];
      for(unsigned int i = 0; i < 8; i++) {
        Lmat[cNum][eType][i] = new double[8];
        if((rank % THOUSAND) == 0) {
          for(unsigned int j = 0; j < 8; j++) {
            fscanf(infile,"%lf",&(Lmat[cNum][eType][i][j]));
          }
        }
      }
    }
  }

  if((rank % THOUSAND) == 0) {
    fclose(infile);
  }

  double* tmpMat = new double[9216];
  assert(tmpMat);

  if((rank % THOUSAND) == 0) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 8; i++) {
          for(unsigned int j = 0; j < 8; j++) {
            tmpMat[ctr] = Lmat[cNum][eType][i][j];
            ctr++;
          }
        }
      }
    }
  }

  par::Mpi_Bcast<double>(tmpMat, 9216, 0, newComm);

  if((rank % THOUSAND) != 0) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 8; i++) {
          for(unsigned int j = 0; j < 8; j++) {
            Lmat[cNum][eType][i][j] = tmpMat[ctr];
            ctr++;
          }
        }
      }
    }
  }

  delete [] tmpMat;
  tmpMat = NULL;
  return 1;
}//end of function


int createLmat_Type1(double ****& Lmat) {
  FILE* infile;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(!rank) {
    char fname[250];
    sprintf(fname,"LapType2Stencils.inp");
    infile = fopen(fname,"r");
    if(!infile) {
      std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
      assert(false);
    }
  }

  typedef double* doublePtr;
  typedef doublePtr* double2Ptr;
  typedef double2Ptr* double3Ptr;

  Lmat = new double3Ptr[8];
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    Lmat[cNum] = new double2Ptr[18];
    for(unsigned int eType = 0; eType < 18; eType++) {
      Lmat[cNum][eType] = new doublePtr[8];
      for(unsigned int i = 0; i < 8; i++) {
        Lmat[cNum][eType][i] = new double[8];
        if(!rank) {
          for(unsigned int j = 0; j < 8; j++) {
            fscanf(infile,"%lf",&(Lmat[cNum][eType][i][j]));
          }
        }
      }
    }
  }

  if(!rank) {
    fclose(infile);
  }

  double* tmpMat = new double[9216];
  assert(tmpMat);

  if(!rank) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 8; i++) {
          for(unsigned int j = 0; j < 8; j++) {
            tmpMat[ctr] = Lmat[cNum][eType][i][j];
            ctr++;
          }
        }
      }
    }
  }

  par::Mpi_Bcast<double>(tmpMat,9216, 0, MPI_COMM_WORLD);

  if(rank) {
    unsigned int ctr = 0;
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      for(unsigned int eType = 0; eType < 18; eType++) {
        for(unsigned int i = 0; i < 8; i++) {
          for(unsigned int j = 0; j < 8; j++) {
            Lmat[cNum][eType][i][j] = tmpMat[ctr];
            ctr++;
          }
        }
      }
    }
  }

  delete [] tmpMat;
  tmpMat = NULL;
  return 1;
}//end of function


int destroyLmat(double****& Lmat ) {
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    for(unsigned int eType = 0; eType < 18; eType++) {
      for(unsigned int i = 0; i < 8; i++) {
        delete [] (Lmat[cNum][eType][i]);
        Lmat[cNum][eType][i] = NULL;
      }
      delete [] (Lmat[cNum][eType]);
      Lmat[cNum][eType] = NULL;
    }
    delete [] (Lmat[cNum]);
    Lmat[cNum] = NULL;
  }

  delete [] Lmat;
  Lmat = NULL;
  return 1;
}//end of function


int destroyGDmat(double****& GDmat) {
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    for(unsigned int eType = 0; eType < 18; eType++) {
      for(unsigned int i = 0; i < 24; i++) {
        delete [] (GDmat[cNum][eType][i]);
        GDmat[cNum][eType][i] = NULL;
      }
      delete [] (GDmat[cNum][eType]);
      GDmat[cNum][eType] = NULL;
    }
    delete [] (GDmat[cNum]);
    GDmat[cNum] = NULL;
  }

  delete [] GDmat;
  GDmat = NULL;
  return 1;
}//end fn.


int destroyPhimat(double******& Phimat, int numGpts) {
  for(unsigned int cNum = 0; cNum < 8; cNum++) {
    for(unsigned int eType = 0; eType < 18; eType++) {
      for(unsigned int i = 0; i < 8; i++) {
        for(unsigned int j = 0; j < numGpts; j++) {
          for(unsigned int k = 0; k < numGpts; k++) {
            delete [] (Phimat[cNum][eType][i][j][k]);
            Phimat[cNum][eType][i][j][k] = NULL;
          }
          delete [] (Phimat[cNum][eType][i][j]);
          Phimat[cNum][eType][i][j] = NULL;
        }
        delete [] (Phimat[cNum][eType][i]);
        Phimat[cNum][eType][i] = NULL;
      }
      delete [] (Phimat[cNum][eType]);
      Phimat[cNum][eType] = NULL;
    }
    delete [] (Phimat[cNum]);
    Phimat[cNum] = NULL;
  }

  delete [] Phimat;
  Phimat = NULL;
  return 1;
}

void createGaussPtsAndWts(double*& gPts, double*& gWts, int numGpts) {
  gPts = new double[numGpts];
  gWts = new double[numGpts];

  if(numGpts == 3) {
    //3-pt rule
    gWts[0] = 0.88888889;  gWts[1] = 0.555555556;  gWts[2] = 0.555555556;
    gPts[0] = 0.0;  gPts[1] = 0.77459667;  gPts[2] = -0.77459667;
  } else if(numGpts == 4) {
    //4-pt rule
    gWts[0] = 0.65214515;  gWts[1] = 0.65214515;
    gWts[2] = 0.34785485; gWts[3] = 0.34785485;  
    gPts[0] = 0.33998104;  gPts[1] = -0.33998104;
    gPts[2] = 0.86113631; gPts[3] = -0.86113631;
  } else if(numGpts == 5) {
    //5-pt rule
    gWts[0] = 0.568888889;  gWts[1] = 0.47862867;  gWts[2] =  0.47862867;
    gWts[3] = 0.23692689; gWts[4] = 0.23692689;
    gPts[0] = 0.0;  gPts[1] = 0.53846931; gPts[2] = -0.53846931;
    gPts[3] = 0.90617985; gPts[4] = -0.90617985;
  } else if(numGpts == 6) {
    //6-pt rule
    gWts[0] = 0.46791393;  gWts[1] = 0.46791393;  gWts[2] = 0.36076157;
    gWts[3] = 0.36076157; gWts[4] = 0.17132449; gWts[5] = 0.17132449;
    gPts[0] = 0.23861918; gPts[1] = -0.23861918; gPts[2] = 0.66120939;
    gPts[3] = -0.66120939; gPts[4] = 0.93246951; gPts[5] = -0.93246951;
  } else if(numGpts == 7) {
    //7-pt rule
    gWts[0] = 0.41795918;  gWts[1] = 0.38183005; gWts[2] = 0.38183005;
    gWts[3] = 0.27970539;  gWts[4] = 0.27970539; 
    gWts[5] = 0.12948497; gWts[6] = 0.12948497;
    gPts[0] = 0.0;  gPts[1] = 0.40584515;  gPts[2] = -0.40584515;
    gPts[3] = 0.74153119;  gPts[4] = -0.74153119;
    gPts[5] = 0.94910791; gPts[6] = -0.94910791;
  } else  {
    assert(false);
  }
}

void destroyGaussPtsAndWts(double*& gPts, double*& gWts) {
  assert(gPts);
  delete [] gPts;
  gPts = NULL;

  assert(gWts);
  delete [] gWts;
  gWts = NULL;
}




