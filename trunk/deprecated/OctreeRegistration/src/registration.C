
/**
  @file registration.C
  @brief Components of Elastic Registration
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  */

#include "mpi.h"
#include "oda.h"
#include "petscda.h"
#include "petscmat.h"
#include "omg.h"
#include "odaUtils.h"
#include "seqUtils.h"
#include <cmath>
#include <iostream>
#include "registration.h"
#include "regInterpCubic.h"
#include "dbh.h"

extern int hessMultEvent;
extern int hessFinestMultEvent;
extern int createHessContextEvent;
extern int updateHessContextEvent;
extern int elasMultEvent;
extern int evalObjEvent;
extern int evalGradEvent;
extern int createPatchesEvent;
extern int expandPatchesEvent;
extern int meshPatchesEvent;
extern int copyValsToPatchesEvent;
extern int optEvent;

#ifndef CUBE
#define CUBE(a) ((a)*(a)*(a))
#endif

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

#ifndef __PI__
#define __PI__ 3.14159265
#endif

#define DiffX(cArr, psi, eta, gamma) ((cArr[1]) + ((cArr[4])*(eta)) + \
    ((cArr[6])*(gamma)) + ((cArr[7])*(eta)*(gamma)))

#define DiffY(cArr, psi, eta, gamma) ((cArr[2]) + ((cArr[4])*(psi)) + \
    ((cArr[5])*(gamma)) + ((cArr[7])*(psi)*(gamma)))

#define DiffZ(cArr, psi, eta, gamma) ((cArr[3]) + ((cArr[5])*(eta)) + \
    ((cArr[6])*(psi)) + ((cArr[7])*(eta)*(psi)))

namespace ot {
  extern double**** ShapeFnCoeffs;
}

void coarsenPrlImage(DA daf, DA dac, bool cActive, 
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

  assert(imgF.size() == (fnxe*fnye*fnze));

  int npesF;
  MPI_Comm_size(commF, &npesF);

  //First Align imgF with dac partition

  //Do Not Free lx, ly, lz. They are managed by DA
  PetscInt* clx = NULL;
  PetscInt* cly = NULL;
  PetscInt* clz = NULL;
  PetscInt cnpx, cnpy, cnpz;
  if(!rank) {
    DAGetOwnershipRange(dac, &clx, &cly, &clz);
    DAGetInfo(dac, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &cnpx, &cnpy, &cnpz,	
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  }

  par::Mpi_Bcast<PetscInt>(&cnpx, 1, 0, commF);
  par::Mpi_Bcast<PetscInt>(&cnpy, 1, 0, commF);
  par::Mpi_Bcast<PetscInt>(&cnpz, 1, 0, commF);

  if(rank) {
    clx = new PetscInt[cnpx];
    assert(clx);
    cly = new PetscInt[cnpy];
    assert(cly);
    clz = new PetscInt[cnpz];
    assert(clz);
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

  if(rank) {
    delete[] clx;
    delete[] cly;
    delete[] clz;
    clx = NULL;
    cly = NULL;
    clz = NULL;
  }

  int* sendCnts = new int[npesF];
  assert(sendCnts);

  int* part = new int[(fnxe*fnye*fnze)];
  assert(part);

  for(int i = 0; i < npesF; i++) {
    sendCnts[i] = 0;
  }//end for i

  int ptCnt = 0;
  for(int k = fzs; k < (fzs + fnze); k++) {
    for(int j = fys; j < (fys + fnye); j++) {
      for(int i = fxs; i < (fxs + fnxe); i++) {
        double xPt = (static_cast<double>(i))*hf;
        double yPt = (static_cast<double>(j))*hf;
        double zPt = (static_cast<double>(k))*hf;
        unsigned int xRes, yRes, zRes;
        seq::maxLowerBound<double>(scanLx, xPt, xRes, 0, 0);
        seq::maxLowerBound<double>(scanLy, yPt, yRes, 0, 0);
        seq::maxLowerBound<double>(scanLz, zPt, zRes, 0, 0);
        part[ptCnt] = (((zRes*cnpy) + yRes)*cnpx) + xRes;
        assert(part[ptCnt] < npesF);
        sendCnts[part[ptCnt]] += 4;
        ptCnt++;
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

  double* xyzSendVals = new double[4*(fnxe*fnye*fnze)];
  assert(xyzSendVals);

  ptCnt = 0;
  for(int k = fzs; k < (fzs + fnze); k++) {
    for(int j = fys; j < (fys + fnye); j++) {
      for(int i = fxs; i < (fxs + fnxe); i++) {
        double xPt = (static_cast<double>(i))*hf;
        double yPt = (static_cast<double>(j))*hf;
        double zPt = (static_cast<double>(k))*hf;
        int idx = sendOffsets[part[ptCnt]] + tmpSendCnts[part[ptCnt]];
        tmpSendCnts[part[ptCnt]] += 4;
        xyzSendVals[idx] = xPt;
        xyzSendVals[idx + 1] = yPt;
        xyzSendVals[idx + 2] = zPt;
        xyzSendVals[idx + 3] = imgF[ptCnt];
        ptCnt++;
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

  int numRecvPts = (recvOffsets[npesF - 1] + recvCnts[npesF - 1])/4;

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

  imgC.resize(cnxe*cnye*cnze);
  std::vector<double> imgFlocal(8*cnxe*cnye*cnze);

  assert(numRecvPts == (8*cnxe*cnye*cnze));

  for(int i = 0; i < numRecvPts; i++) {
    double xPt = xyzRecvVals[(4*i)];
    double yPt = xyzRecvVals[(4*i) + 1];
    double zPt = xyzRecvVals[(4*i) + 2];
    double val = xyzRecvVals[(4*i) + 3];
    int xi = (static_cast<int>(xPt/hf)) - (2*cxs);
    int yi = (static_cast<int>(yPt/hf)) - (2*cys);
    int zi = (static_cast<int>(zPt/hf)) - (2*czs);
    int idx = (((zi*(2*cnye)) + yi)*(2*cnxe)) + xi;
    assert(idx < imgFlocal.size());
    imgFlocal[idx] = val;
  }//end for i

  assert(xyzRecvVals);
  delete [] xyzRecvVals;
  xyzRecvVals = NULL;

  //Coarsen Locally
  for(int k = 0; k < cnze; k++) {
    for(int j = 0; j < cnye; j++) {
      for(int i = 0; i < cnxe; i++) {
        imgC[(((k*cnye) + j)*cnxe) + i] = 0.125*(
            imgFlocal[((((2*k)*(2*cnye)) + (2*j))*(2*cnxe)) + (2*i)] +
            imgFlocal[((((2*k)*(2*cnye)) + (2*j))*(2*cnxe)) + ((2*i) + 1)] +
            imgFlocal[((((2*k)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + (2*i)] +
            imgFlocal[((((2*k)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + ((2*i) + 1)] +
            imgFlocal[(((((2*k) + 1)*(2*cnye)) + (2*j))*(2*cnxe)) + (2*i)] +
            imgFlocal[(((((2*k) + 1)*(2*cnye)) + (2*j))*(2*cnxe)) + ((2*i) + 1)] +
            imgFlocal[(((((2*k) + 1)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + (2*i)] +
            imgFlocal[(((((2*k) + 1)*(2*cnye)) + ((2*j) + 1))*(2*cnxe)) + ((2*i) + 1)]);
      }//end for i
    }//end for j
  }//end for k

}


void createImgNodalNatural(DA da, const std::vector<double>& sigElemImg,
    const std::vector<double>& tauElemImg, Vec & sigNatural, Vec & tauNatural) {

  DACreateNaturalVector(da, &tauNatural);
  VecDuplicate(tauNatural, &sigNatural);

  Vec sigGlobal;
  Vec tauGlobal;
  DACreateGlobalVector(da, &tauGlobal);
  VecDuplicate(tauGlobal, &sigGlobal);

  //Number of nodes is 1 more than the number of elements
  PetscInt Ne;
  DAGetInfo(da, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,	
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  Ne--;

  VecZeroEntries(sigGlobal);
  VecZeroEntries(tauGlobal);

  PetscScalar*** sigGlobalArr;
  PetscScalar*** tauGlobalArr;
  DAVecGetArray(da, sigGlobal, &sigGlobalArr);
  DAVecGetArray(da, tauGlobal, &tauGlobalArr);

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

  assert(sigElemImg.size() == (nxe*nye*nze));
  assert(tauElemImg.size() == (nxe*nye*nze));

  int ptCnt = 0;
  for(int k = zs; k < zs + nze; k++) {
    for(int j = ys; j < ys + nye; j++) {
      for(int i = xs; i < xs + nxe; i++) {
        sigGlobalArr[k][j][i] = sigElemImg[ptCnt];
        tauGlobalArr[k][j][i] = tauElemImg[ptCnt];
        ptCnt++;
      }//end for i
    }//end for j
  }//end for k

  DAVecRestoreArray(da, sigGlobal, &sigGlobalArr);
  DAVecRestoreArray(da, tauGlobal, &tauGlobalArr);

  DAGlobalToNaturalBegin(da, tauGlobal, INSERT_VALUES, tauNatural);
  DAGlobalToNaturalEnd(da, tauGlobal, INSERT_VALUES, tauNatural);

  DAGlobalToNaturalBegin(da, sigGlobal, INSERT_VALUES, sigNatural);
  DAGlobalToNaturalEnd(da, sigGlobal, INSERT_VALUES, sigNatural);

  VecDestroy(sigGlobal);
  VecDestroy(tauGlobal);

}


void newProcessImgNatural(DA da1dof, DA da3dof, int Ne, Vec sigNatural, Vec tauNatural,
    std::vector<std::vector<double> > &  sigGlobal, std::vector<std::vector<double> > & gradSigGlobal,
    std::vector<std::vector<double> > & tauGlobal, std::vector<std::vector<double> > & gradTauGlobal,
    std::vector<double> & sigElemental, std::vector<double> & tauElemental) {

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

  PetscScalar*** sigGlobalArr;
  PetscScalar*** tauGlobalArr;
  DAVecGetArray(da1dof, sigGvec, &sigGlobalArr);
  DAVecGetArray(da1dof, tauGvec, &tauGlobalArr);

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

  double h = 1.0/static_cast<double>(Ne);

  sigElemental.clear();
  tauElemental.clear();
  for(int k = zs; k < zs + nze; k++) {
    for(int j = ys; j < ys + nye; j++) {
      for(int i = xs; i < xs + nxe; i++) {
        double xPos = static_cast<double>(i)*h;
        double yPos = static_cast<double>(j)*h;
        double zPos = static_cast<double>(k)*h;
        double valSig = sigGlobalArr[k][j][i];
        double valTau = tauGlobalArr[k][j][i];
        sigElemental.push_back(valSig);
        tauElemental.push_back(valTau);
      }
    }
  }

  DAVecRestoreArray(da1dof, sigGvec, &sigGlobalArr);
  DAVecRestoreArray(da1dof, tauGvec, &tauGlobalArr);

  computeFDgradient(da1dof, da3dof, sigGvec, gradSigGvec);
  computeFDgradient(da1dof, da3dof, tauGvec, gradTauGvec);

  PetscInt gVecSz;
  VecGetLocalSize(sigGvec, &gVecSz);

  assert(gVecSz == (nx*ny*nz));

  sigGlobal.resize(1);
  gradSigGlobal.resize(1);
  tauGlobal.resize(1);
  gradTauGlobal.resize(1);
  sigGlobal[0].resize(nx*ny*nz);
  gradSigGlobal[0].resize(3*nx*ny*nz);
  tauGlobal[0].resize(nx*ny*nz);
  gradTauGlobal[0].resize(3*nx*ny*nz);

  PetscScalar* sigGlinArr;
  PetscScalar* tauGlinArr;
  PetscScalar* gradSigGlinArr;
  PetscScalar* gradTauGlinArr;

  VecGetArray(sigGvec, &sigGlinArr);
  VecGetArray(gradSigGvec, &gradSigGlinArr);

  VecGetArray(tauGvec, &tauGlinArr);
  VecGetArray(gradTauGvec, &gradTauGlinArr);

  for(int i = 0; i < (nx*ny*nz); i++) {
    sigGlobal[0][i] = sigGlinArr[i];
    tauGlobal[0][i] = tauGlinArr[i];
    for(int j = 0; j < 3; j++) {
      gradSigGlobal[0][(3*i) + j] = gradSigGlinArr[(3*i) + j];
      gradTauGlobal[0][(3*i) + j] = gradTauGlinArr[(3*i) + j];
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


void copyValuesToImagePatches(DA dar, 
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<std::vector<double> >& sigGlobal,
    const std::vector<std::vector<double> >& gradSigGlobal,
    const std::vector<std::vector<double> >& tauGlobal, 
    const std::vector<std::vector<double> >& gradTauGlobal,
    std::vector<std::vector<double> >& sigLocal,
    std::vector<std::vector<double> >& gradSigLocal,
    std::vector<std::vector<double> >& tauLocal,
    std::vector<std::vector<double> >& gradTauLocal) {

  PetscLogEventBegin(copyValsToPatchesEvent, 0, 0, 0, 0);

  //Active processors for the Octree mesh must be a subset or equal to that for
  //the DA mesh

  MPI_Comm comm;
  PetscObjectGetComm((PetscObject)dar, &comm);

  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  PetscInt npx, npy, npz;
  PetscInt Ne;

  //Do Not Free lx, ly, lz. They are managed by DA

  PetscInt* lx = NULL;
  PetscInt* ly = NULL;
  PetscInt* lz = NULL;

  //Number of nodes is 1 more than the number of elements
  DAGetOwnershipRange(dar, &lx, &ly, &lz);
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

  unsigned int numSendPts = imgPatches.size();
  int* sendCnts = new int[npes];
  assert(sendCnts);
  int* part = NULL;
  if(numSendPts) {
    part = new int[numSendPts];
    assert(part);
  }

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
  }//end for i

  unsigned int maxDepth;
  double hFac;

  if(!(imgPatches.empty())) {
    maxDepth = imgPatches[0].getMaxDepth();
    hFac = 1.0/static_cast<double>(1u << (maxDepth - 1));
  }

  for(int i = 0; i < numSendPts; i++) {
    double xPt = hFac*(static_cast<double>(imgPatches[i].getX()));
    double yPt = hFac*(static_cast<double>(imgPatches[i].getY()));
    double zPt = hFac*(static_cast<double>(imgPatches[i].getZ()));
    unsigned int xRes, yRes, zRes;
    seq::maxLowerBound<double>(scanLx, xPt, xRes, 0, 0);
    seq::maxLowerBound<double>(scanLy, yPt, yRes, 0, 0);
    seq::maxLowerBound<double>(scanLz, zPt, zRes, 0, 0);
    part[i] = (((zRes*npy) + yRes)*npx) + xRes;
    assert(part[i] < npes);
    sendCnts[part[i]] += 3;
  }//end for i

  scanLx.clear();
  scanLy.clear();
  scanLz.clear();  

  int* recvCnts = new int[npes];
  assert(recvCnts);

  par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, comm);

  int totalRecv = 0;
  for(int i = 0; i < npes; i++) {
    totalRecv += recvCnts[i];
  }

  int* sendOffsets = new int[npes];
  assert(sendOffsets);

  int* recvOffsets = new int[npes];
  assert(recvOffsets);

  sendOffsets[0] = 0;
  recvOffsets[0] = 0;
  for(int i = 1; i < npes; i++) {
    sendOffsets[i] = sendOffsets[i - 1] + sendCnts[i - 1];
    recvOffsets[i] = recvOffsets[i - 1] + recvCnts[i - 1];
  }//end for i

  int* commMap = NULL;
  if(numSendPts) {
    commMap = new int[numSendPts];  
    assert(commMap);
  }

  int totalSend = (3*numSendPts);

  double* xyzSendVals = NULL;
  if(totalSend) {
    xyzSendVals = new double[totalSend];
    assert(xyzSendVals);
  }

  int* tmpSendCnts = new int[npes];  
  assert(tmpSendCnts);
  for(int i = 0; i < npes; i++) {
    tmpSendCnts[i] = 0;
  }

  for(int i = 0; i < numSendPts; i++) {
    double xPt = hFac*(static_cast<double>(imgPatches[i].getX()));
    double yPt = hFac*(static_cast<double>(imgPatches[i].getY()));
    double zPt = hFac*(static_cast<double>(imgPatches[i].getZ()));

    int idx = sendOffsets[part[i]] + tmpSendCnts[part[i]];

    assert(part[i] < npes);
    tmpSendCnts[part[i]] += 3;

    xyzSendVals[idx] = xPt;
    xyzSendVals[idx + 1] = yPt;
    xyzSendVals[idx + 2] = zPt;

    commMap[i] = idx/3;
  }//end for i

  assert(tmpSendCnts);
  delete [] tmpSendCnts;
  tmpSendCnts = NULL;

  if(part) {
    delete [] part;
    part = NULL;
  }

  double* xyzRecvVals = NULL;
  if(totalRecv) {
    xyzRecvVals = new double[totalRecv];
    assert(xyzRecvVals);
  }

  par::Mpi_Alltoallv_sparse<double>(xyzSendVals, sendCnts, sendOffsets,
      xyzRecvVals, recvCnts, recvOffsets, comm);

  if(xyzSendVals) {
    delete [] xyzSendVals;
    xyzSendVals = NULL;
  }

  PetscInt xs, ys, zs, nx, ny, nz;
  DAGetCorners(dar, &xs, &ys, &zs, &nx, &ny, &nz);

  unsigned int numRecvPts = totalRecv/3;

  unsigned int numImages = sigGlobal.size();
  std::vector<std::vector<double> > tmpSendSigVals(numImages);
  std::vector<std::vector<double> > tmpSendTauVals(numImages);
  std::vector<std::vector<double> > tmpSendGradSigVals(numImages);
  std::vector<std::vector<double> > tmpSendGradTauVals(numImages);

  assert(numImages);
  for(int i = 0; i < numImages; i++) {
    tmpSendSigVals[i].resize(numRecvPts);
    tmpSendTauVals[i].resize(numRecvPts);
    tmpSendGradSigVals[i].resize(3*numRecvPts);
    tmpSendGradTauVals[i].resize(3*numRecvPts);
  }//end for i

  for(unsigned int i = 0; i < numRecvPts; i++) {
    double xPt = xyzRecvVals[(3*i)];
    double yPt = xyzRecvVals[(3*i) + 1];
    double zPt = xyzRecvVals[(3*i) + 2];

    unsigned int xi = static_cast<int>(xPt/h) - xs;
    unsigned int yi = static_cast<int>(yPt/h) - ys;
    unsigned int zi = static_cast<int>(zPt/h) - zs;

    assert(xi < nx);
    assert(yi < ny);
    assert(zi < nz);

    unsigned int globArrIdx = ((((zi*ny) + yi)*nx) + xi);

    for(int j = 0; j < numImages; j++) {
      assert( globArrIdx < sigGlobal[j].size() );
      assert( globArrIdx < tauGlobal[j].size() );
      tmpSendSigVals[j][i] = sigGlobal[j][globArrIdx];
      tmpSendTauVals[j][i] = tauGlobal[j][globArrIdx];
      for(int d = 0; d < 3; d++) {
        assert( ((3*globArrIdx) + d) < gradSigGlobal[j].size() );
        assert( ((3*globArrIdx) + d) < gradTauGlobal[j].size() );
        tmpSendGradSigVals[j][(3*i) + d] = gradSigGlobal[j][(3*globArrIdx) + d];
        tmpSendGradTauVals[j][(3*i) + d] = gradTauGlobal[j][(3*globArrIdx) + d];
      }//end for d
    }//end for j
  }//end for i

  if(xyzRecvVals) {
    delete [] xyzRecvVals;
    xyzRecvVals = NULL;
  }

  std::vector<std::vector<double> > tmpRecvSigVals(numImages);
  std::vector<std::vector<double> > tmpRecvTauVals(numImages);
  std::vector<std::vector<double> > tmpRecvGradSigVals(numImages);
  std::vector<std::vector<double> > tmpRecvGradTauVals(numImages);

  for(int i = 0; i < numImages; i++) {
    tmpRecvSigVals[i].resize(numSendPts);
    tmpRecvTauVals[i].resize(numSendPts);
    tmpRecvGradSigVals[i].resize(3*numSendPts);
    tmpRecvGradTauVals[i].resize(3*numSendPts);
  }//end for i

  for(int i = 0; i < numImages; i++) {
    double* tmpSendGradSigValsPtr = NULL;
    double* tmpRecvGradSigValsPtr = NULL;
    double* tmpSendGradTauValsPtr = NULL;
    double* tmpRecvGradTauValsPtr = NULL;
    if(!(tmpSendGradSigVals[i].empty())) {
      tmpSendGradSigValsPtr = (&(*((tmpSendGradSigVals[i]).begin())));
    }
    if(!(tmpRecvGradSigVals[i].empty())) {
      tmpRecvGradSigValsPtr = (&(*((tmpRecvGradSigVals[i]).begin())));
    }
    if(!(tmpSendGradTauVals[i].empty())) {
      tmpSendGradTauValsPtr = (&(*((tmpSendGradTauVals[i]).begin())));
    }
    if(!(tmpRecvGradTauVals[i].empty())) {
      tmpRecvGradTauValsPtr = (&(*((tmpRecvGradTauVals[i]).begin())));
    }
    par::Mpi_Alltoallv_sparse<double>( tmpSendGradSigValsPtr, recvCnts, recvOffsets,
        tmpRecvGradSigValsPtr, sendCnts, sendOffsets, comm);
    par::Mpi_Alltoallv_sparse<double>( tmpSendGradTauValsPtr, recvCnts, recvOffsets,
        tmpRecvGradTauValsPtr, sendCnts, sendOffsets, comm);
  }//end for i

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = sendCnts[i]/3;
    recvCnts[i] = recvCnts[i]/3;
    sendOffsets[i] = sendOffsets[i]/3;
    recvOffsets[i] = recvOffsets[i]/3;
  }//end for i

  for(int i = 0; i < numImages; i++) {
    double* tmpSendSigValsPtr = NULL;
    double* tmpRecvSigValsPtr = NULL;
    double* tmpSendTauValsPtr = NULL;
    double* tmpRecvTauValsPtr = NULL;
    if(!(tmpSendSigVals[i].empty())) {
      tmpSendSigValsPtr = (&(*((tmpSendSigVals[i]).begin())));
    }
    if(!(tmpRecvSigVals[i].empty())) {
      tmpRecvSigValsPtr = (&(*((tmpRecvSigVals[i]).begin())));
    }
    if(!(tmpSendTauVals[i].empty())) {
      tmpSendTauValsPtr = (&(*((tmpSendTauVals[i]).begin())));
    }
    if(!(tmpRecvTauVals[i].empty())) {
      tmpRecvTauValsPtr = (&(*((tmpRecvTauVals[i]).begin())));
    } 
    par::Mpi_Alltoallv_sparse<double>( tmpSendSigValsPtr, recvCnts, recvOffsets,
        tmpRecvSigValsPtr, sendCnts, sendOffsets, comm);
    par::Mpi_Alltoallv_sparse<double>( tmpSendTauValsPtr, recvCnts, recvOffsets,
        tmpRecvTauValsPtr, sendCnts, sendOffsets, comm);
  }//end for i

  tmpSendSigVals.clear();
  tmpSendTauVals.clear();
  tmpSendGradSigVals.clear();
  tmpSendGradTauVals.clear();

  assert(sendCnts);
  delete [] sendCnts;
  sendCnts = NULL;

  assert(recvCnts);
  delete [] recvCnts;
  recvCnts = NULL;

  assert(sendOffsets);
  delete [] sendOffsets;
  sendOffsets = NULL;

  assert(recvOffsets);
  delete [] recvOffsets;
  recvOffsets = NULL;

  sigLocal.resize(numImages);
  tauLocal.resize(numImages);
  gradSigLocal.resize(numImages);
  gradTauLocal.resize(numImages);

  for(int i = 0; i < numImages; i++) {
    sigLocal[i].resize(numSendPts);
    tauLocal[i].resize(numSendPts);
    gradSigLocal[i].resize(3*numSendPts);
    gradTauLocal[i].resize(3*numSendPts);
  }//end for i

  for(int j = 0; j < numImages; j++) {
    for(int i = 0; i < numSendPts; i++) {
      assert(commMap[i] < tmpRecvSigVals[j].size());
      assert(commMap[i] < tmpRecvTauVals[j].size());
      sigLocal[j][i] = tmpRecvSigVals[j][commMap[i]];
      tauLocal[j][i] = tmpRecvTauVals[j][commMap[i]];
      for(int d = 0; d < 3; d++) {
        gradSigLocal[j][(3*i) + d] = tmpRecvGradSigVals[j][(3*(commMap[i])) + d];
        gradTauLocal[j][(3*i) + d] = tmpRecvGradTauVals[j][(3*(commMap[i])) + d];
      }//end for d
    }//end for i
  }//end for j

  if(commMap) {
    delete [] commMap;
    commMap = NULL;
  }

  PetscLogEventEnd(copyValsToPatchesEvent, 0, 0, 0, 0);

}

void meshImagePatches(std::vector<ot::TreeNode>& imgPatches,
    std::vector<unsigned int>& mesh) {

  PetscLogEventBegin(meshPatchesEvent, 0, 0, 0, 0);

  mesh.clear();

  if(!(imgPatches.empty())) {
    //This is the maxDepth used inside DA
    unsigned int maxDepth = imgPatches[0].getMaxDepth();
    unsigned int regLev = imgPatches[0].getLevel();
    unsigned int hRg = (1u << (maxDepth - regLev));
    unsigned int maxPos = (1u << (maxDepth - 1));

    //We only want to mesh true elements, not positive boundaries
    unsigned int numElems = 0;
    for(int i = 0 ; i < imgPatches.size(); i++) {
      unsigned int myX = imgPatches[i].getX();
      unsigned int myY = imgPatches[i].getY();
      unsigned int myZ = imgPatches[i].getZ();
      if( (myX < maxPos) && (myY < maxPos) && (myZ < maxPos) ) {
        numElems++; 
      } else {
        break; 
      }
    }//end for i

    mesh.resize(8*numElems);

    for(unsigned int i = 0; i < numElems; i++) {
      unsigned int myX = imgPatches[i].getX();
      unsigned int myY = imgPatches[i].getY();
      unsigned int myZ = imgPatches[i].getZ();
      for(int j = 0; j < 8; j++) {
        int xbit = (j%2);
        int ybit = ((j/2)%2);
        int zbit = (j/4);
        unsigned int x = myX + (xbit*hRg);
        unsigned int y = myY + (ybit*hRg);
        unsigned int z = myZ + (zbit*hRg);
        ot::TreeNode searchNode(x, y, z, regLev, 3, maxDepth);
        unsigned int retIdx;
        bool found = seq::BinarySearch<ot::TreeNode>(&(*(imgPatches.begin())),
            imgPatches.size(), searchNode, &retIdx);
        if(found) {
          mesh[(8*i) + j] = retIdx;
        } else {
          //Some elements are only vertices, so we will not find their vertices 
          mesh[(8*i) + j] = static_cast<unsigned int>(-1);
        }
      }//end for j
    }//end for i

  }//end if active

  PetscLogEventEnd(meshPatchesEvent, 0, 0, 0, 0);
}


void expandImagePatches(int Ne, double width,
    const std::vector<ot::TreeNode>& blocks,
    std::vector<ot::TreeNode>& imgPatches) {

  PetscLogEventBegin(expandPatchesEvent, 0, 0, 0, 0);

  if(!(imgPatches.empty())) {
    //This is the maxDepth used inside DA
    unsigned int maxDepth = imgPatches[0].getMaxDepth();
    assert(!(blocks.empty()));
    assert(blocks[0].getMaxDepth() == maxDepth);

    unsigned int regLev = (binOp::fastLog2(Ne)) + 1;

    assert(imgPatches[0].getLevel() == regLev);

    unsigned int maxPos = (1u << (maxDepth - 1));
    double hInvFac = static_cast<double>(maxPos);
    double hFac = 1.0/hInvFac;

    unsigned int numPad = static_cast<unsigned int>(ceil(width*static_cast<double>(Ne)));
    unsigned int hRg = (1u << (maxDepth - regLev));

    //At this point, imgPatches is a complete globally sorted linear octree (actually
    //regular grid). There are no overlaps across processors, i.e. it is a
    //partition of unity.
    //The boundaries controlled by this processor 
    ot::TreeNode minOct = imgPatches[0];
    ot::TreeNode maxOct = (imgPatches[imgPatches.size() - 1]).getDLD();

    //First select octants whose insulation is not already contained within this
    //processor
    std::vector<ot::TreeNode> selectedOcts;
    for(int i = 0; i < blocks.size(); i++) {
      if(blocks[i] > maxOct) {
        //Blocks may include positive boundaries, which we must avoid here 
        assert( (blocks[i].minX() >= maxPos) ||
            (blocks[i].minY() >= maxPos) ||
            (blocks[i].minZ() >= maxPos) );
        break;
      }
      double myMinX = (static_cast<double>(blocks[i].minX())*hFac); 
      double myMinY = (static_cast<double>(blocks[i].minY())*hFac); 
      double myMinZ = (static_cast<double>(blocks[i].minZ())*hFac); 

      double myMaxX = (static_cast<double>(blocks[i].maxX())*hFac); 
      double myMaxY = (static_cast<double>(blocks[i].maxY())*hFac); 
      double myMaxZ = (static_cast<double>(blocks[i].maxZ())*hFac); 

      double minX = myMinX - width; 
      double minY = myMinY - width; 
      double minZ = myMinZ - width; 

      double maxX = myMaxX + width; 
      double maxY = myMaxY + width;     
      double maxZ = myMaxZ + width; 

      if(minX < 0) {
        minX = 0;
      }
      if(minY < 0) {
        minY = 0;
      }
      if(minZ < 0) {
        minZ = 0;
      }

      if(maxX > 1) {
        maxX = 1;
      }
      if(maxY > 1) {
        maxY = 1;
      }
      if(maxZ > 1) {
        maxZ = 1;
      }

      unsigned int minXint = static_cast<unsigned int>(minX*hInvFac);
      unsigned int minYint = static_cast<unsigned int>(minY*hInvFac);
      unsigned int minZint = static_cast<unsigned int>(minZ*hInvFac);

      unsigned int maxXint = static_cast<unsigned int>(maxX*hInvFac) - 1;
      unsigned int maxYint = static_cast<unsigned int>(maxY*hInvFac) - 1;
      unsigned int maxZint = static_cast<unsigned int>(maxZ*hInvFac) - 1;

      ot::TreeNode negOct(minXint, minYint, minZint, maxDepth, 3, maxDepth);
      ot::TreeNode posOct(maxXint, maxYint, maxZint, maxDepth, 3, maxDepth);

      if( (negOct < minOct) || (posOct > maxOct) ) {
        selectedOcts.push_back(blocks[i]);
      }

    }//end for i

    unsigned int imgPatchInitSz = imgPatches.size();

    std::vector<ot::TreeNode> extraOcts;
    std::vector<ot::TreeNode> tmpExtras;
    for(int i = 0; i < selectedOcts.size(); i++) {
      unsigned int xs = selectedOcts[i].minX();
      unsigned int ys = selectedOcts[i].minY();
      unsigned int zs = selectedOcts[i].minZ();

      unsigned int xe = selectedOcts[i].maxX();
      unsigned int ye = selectedOcts[i].maxY();
      unsigned int ze = selectedOcts[i].maxZ();

      for(int j = 0; j < numPad; j++) {
        if(xs >= hRg) {
          xs = xs - hRg;
        } else {
          break;
        }
      }

      for(int j = 0; j < numPad; j++) {
        if(ys >= hRg) {
          ys = ys - hRg;
        } else {
          break;
        }
      }

      for(int j = 0; j < numPad; j++) {
        if(zs >= hRg) {
          zs = zs - hRg;
        } else {
          break;
        }
      }

      for(int j = 0; j < numPad; j++) {
        if((xe + hRg) <= maxPos) {
          xe = xe + hRg;
        } else {
          break;
        }
      }

      for(int j = 0; j < numPad; j++) {
        if((ye + hRg) <= maxPos) {
          ye = ye + hRg;
        } else {
          break;
        }
      }

      for(int j = 0; j < numPad; j++) {
        if((ze + hRg) <= maxPos) {
          ze = ze + hRg;
        } else {
          break;
        }
      }

      for(unsigned int z = zs; z < ze; z += hRg) {
        for(unsigned int y = ys; y < ye; y += hRg) {
          for(unsigned int x = xs; x < xe; x += hRg) {
            ot::TreeNode tmpNode(x, y, z, regLev, 3, maxDepth);
            if( (tmpNode < minOct) || (tmpNode > maxOct) ) {
              extraOcts.push_back(tmpNode);
            }
          }//end for x
        }//end for y
      }//end for z

      if(extraOcts.size() > imgPatches.size()) {
        tmpExtras.insert(tmpExtras.end(),  extraOcts.begin(), extraOcts.end());
        extraOcts.clear();
        seq::makeVectorUnique<ot::TreeNode>(tmpExtras, false);
      }

    }//end for i

    selectedOcts.clear();

    if(!(extraOcts.empty())) {
      tmpExtras.insert(tmpExtras.end(),  extraOcts.begin(), extraOcts.end());
      extraOcts.clear();
      seq::makeVectorUnique<ot::TreeNode>(tmpExtras, false);
    }

    if(!(tmpExtras.empty())) {
      imgPatches.insert(imgPatches.end(), tmpExtras.begin(), tmpExtras.end());
      tmpExtras.clear();
      seq::makeVectorUnique<ot::TreeNode>(imgPatches, false);
    }

    //Elements to Vertices...
    //We search and add missing vertices.
    //Note, we can reuse the search results by doing meshing simultaneously (like
    //in ot::DA::buildNodeList) and then following it up with a mesh-remap to
    //correct the indices. But, it is not worth the effort. So we use a simpler
    //implementation. 
    std::vector<ot::TreeNode> missingOcts;
    for(int i = 0; i < imgPatches.size(); i++) {
      unsigned int myX = imgPatches[i].getX();
      unsigned int myY = imgPatches[i].getY();
      unsigned int myZ = imgPatches[i].getZ();
      for(int j = 0; j < 8; j++) {
        int xbit = (j%2);
        int ybit = ((j/2)%2);
        int zbit = (j/4);
        unsigned int x = myX + (xbit*hRg);
        unsigned int y = myY + (ybit*hRg);
        unsigned int z = myZ + (zbit*hRg);
        ot::TreeNode searchNode(x, y, z, regLev, 3, maxDepth);
        unsigned int retIdx;
        bool found = seq::BinarySearch<ot::TreeNode>(&(*(imgPatches.begin())),
            imgPatches.size(), searchNode, &retIdx);
        if(!found) {
          missingOcts.push_back(searchNode);
        }
      }//end for j
    }//end for i

    seq::makeVectorUnique<ot::TreeNode>(missingOcts, false);

    if(!(missingOcts.empty())) {
      imgPatches.insert(imgPatches.end(), missingOcts.begin(), missingOcts.end());
    }

    sort(imgPatches.begin(), imgPatches.end());

  }//end if active

  PetscLogEventEnd(expandPatchesEvent, 0, 0, 0, 0);
}


void createImagePatches(int Ne, ot::DA* da, 
    std::vector<ot::TreeNode>& imgPatches) {

  PetscLogEventBegin(createPatchesEvent, 0, 0, 0, 0);

  unsigned int regLev = (binOp::fastLog2(Ne)) + 1;

  imgPatches.clear();

  assert(da != NULL);

  if(da->iAmActive()) {
    unsigned int maxDepth = da->getMaxDepth();
    for(da->init<ot::DA_FLAGS::WRITABLE>();
        da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
        da->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = da->getCurrentOffset();
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      imgPatches.push_back(ot::TreeNode(xint, yint, zint, lev, 3, maxDepth));
    }//end WRITABLE

    bool repeatLoop = true;
    while(repeatLoop) {
      repeatLoop = false;
      std::vector<ot::TreeNode> tmpList;
      for(int i = 0; i < imgPatches.size(); i++) {
        if((imgPatches[i].getLevel()) < regLev) {
          imgPatches[i].addChildren(tmpList);
          repeatLoop = true;
        } else {
          assert((imgPatches[i].getLevel()) == regLev);
          tmpList.push_back(imgPatches[i]);
        }
      }
      imgPatches = tmpList;
    }
  }//end if active

  PetscLogEventEnd(createPatchesEvent, 0, 0, 0, 0);
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


void processImgNatural(DA da1dof, DA da3dof, int Ne,
    Vec sigNatural, Vec tauNatural,
    std::vector<std::vector<double> > &  sig, std::vector<std::vector<double> > & gradSig,
    std::vector<std::vector<double> > & tau, std::vector<std::vector<double> > & gradTau,
    std::vector<double> & sigElemental, std::vector<double> & tauElemental) {

  Vec sigGlobal, tauGlobal;
  Vec sigNaturalAll, tauNaturalAll;
  Vec gradSigGlobal, gradTauGlobal;
  Vec gradSigNatural, gradTauNatural;
  Vec gradSigNaturalAll, gradTauNaturalAll;
  VecScatter toLocalAll1dof;
  VecScatter toLocalAll3dof;

  DACreateGlobalVector(da1dof, &tauGlobal);
  DACreateNaturalVector(da3dof, &gradTauNatural);
  DACreateGlobalVector(da3dof, &gradTauGlobal);

  VecScatterCreateToAll(tauNatural, &toLocalAll1dof, &tauNaturalAll);
  VecScatterCreateToAll(gradTauNatural, &toLocalAll3dof, &gradTauNaturalAll);

  VecDuplicate(tauGlobal, &sigGlobal);
  VecDuplicate(gradTauGlobal, &gradSigGlobal);
  VecDuplicate(gradTauNatural, &gradSigNatural);
  VecDuplicate(tauNaturalAll, &sigNaturalAll);
  VecDuplicate(gradTauNaturalAll, &gradSigNaturalAll);

  DANaturalToGlobalBegin(da1dof, tauNatural, INSERT_VALUES, tauGlobal);
  DANaturalToGlobalEnd(da1dof, tauNatural, INSERT_VALUES, tauGlobal);
  DANaturalToGlobalBegin(da1dof, sigNatural, INSERT_VALUES, sigGlobal);
  DANaturalToGlobalEnd(da1dof, sigNatural, INSERT_VALUES, sigGlobal);

  PetscScalar*** tauGlobalArr;
  PetscScalar*** sigGlobalArr;
  DAVecGetArray(da1dof, tauGlobal, &tauGlobalArr);
  DAVecGetArray(da1dof, sigGlobal, &sigGlobalArr);

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

  double h = 1.0/static_cast<double>(Ne);

  tauElemental.clear();
  sigElemental.clear();
  for(int k = zs; k < zs + nze; k++) {
    for(int j = ys; j < ys + nye; j++) {
      for(int i = xs; i < xs + nxe; i++) {
        double xPos = static_cast<double>(i)*h;
        double yPos = static_cast<double>(j)*h;
        double zPos = static_cast<double>(k)*h;
        double valTau = tauGlobalArr[k][j][i];
        double valSig = sigGlobalArr[k][j][i];
        tauElemental.push_back(valTau);
        sigElemental.push_back(valSig);
      }
    }
  }

  DAVecRestoreArray(da1dof, tauGlobal, &tauGlobalArr);
  DAVecRestoreArray(da1dof, sigGlobal, &sigGlobalArr);

  computeFDgradient(da1dof, da3dof, tauGlobal, gradTauGlobal);
  computeFDgradient(da1dof, da3dof, sigGlobal, gradSigGlobal);

  DAGlobalToNaturalBegin(da3dof, gradTauGlobal, INSERT_VALUES, gradTauNatural);
  DAGlobalToNaturalEnd(da3dof, gradTauGlobal, INSERT_VALUES, gradTauNatural);

  DAGlobalToNaturalBegin(da3dof, gradSigGlobal, INSERT_VALUES, gradSigNatural);
  DAGlobalToNaturalEnd(da3dof, gradSigGlobal, INSERT_VALUES, gradSigNatural);

  VecScatterBegin(toLocalAll1dof, tauNatural, tauNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD);
  VecScatterBegin(toLocalAll3dof, gradTauNatural, gradTauNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD );

  VecScatterEnd(toLocalAll1dof, tauNatural, tauNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(toLocalAll3dof, gradTauNatural, gradTauNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(toLocalAll1dof, sigNatural, sigNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD);
  VecScatterBegin(toLocalAll3dof, gradSigNatural, gradSigNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD );

  VecScatterEnd(toLocalAll1dof, sigNatural, sigNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(toLocalAll3dof, gradSigNatural, gradSigNaturalAll,
      INSERT_VALUES, SCATTER_FORWARD);

  VecScatterDestroy(toLocalAll1dof);
  VecScatterDestroy(toLocalAll3dof);

  VecDestroy(gradSigNatural);
  VecDestroy(gradTauNatural);

  VecDestroy(tauGlobal);
  VecDestroy(gradTauGlobal);

  VecDestroy(sigGlobal);
  VecDestroy(gradSigGlobal);

  tau.resize(1);
  gradTau.resize(1);
  sig.resize(1);
  gradSig.resize(1);
  tau[0].resize((Ne + 1)*(Ne + 1)*(Ne + 1));
  gradTau[0].resize(3*(Ne + 1)*(Ne + 1)*(Ne + 1));
  sig[0].resize((Ne + 1)*(Ne + 1)*(Ne + 1));
  gradSig[0].resize(3*(Ne + 1)*(Ne + 1)*(Ne + 1));

  PetscScalar* tauNaturalAllArr;
  PetscScalar* gradTauNaturalAllArr;

  PetscScalar* sigNaturalAllArr;
  PetscScalar* gradSigNaturalAllArr;

  VecGetArray(tauNaturalAll, &tauNaturalAllArr);
  VecGetArray(gradTauNaturalAll, &gradTauNaturalAllArr);

  VecGetArray(sigNaturalAll, &sigNaturalAllArr);
  VecGetArray(gradSigNaturalAll, &gradSigNaturalAllArr);

  for(int i = 0; i < ((Ne + 1)*(Ne + 1)*(Ne + 1)); i++) {
    tau[0][i] = tauNaturalAllArr[i];
    sig[0][i] = sigNaturalAllArr[i];
    for(int j = 0; j < 3; j++) {
      gradTau[0][(3*i) + j] = gradTauNaturalAllArr[(3*i) + j];
      gradSig[0][(3*i) + j] = gradSigNaturalAllArr[(3*i) + j];
    }
  }

  VecRestoreArray(tauNaturalAll, &tauNaturalAllArr);
  VecRestoreArray(gradTauNaturalAll, &gradTauNaturalAllArr);

  VecRestoreArray(sigNaturalAll, &sigNaturalAllArr);
  VecRestoreArray(gradSigNaturalAll, &gradSigNaturalAllArr);

  VecDestroy(tauNaturalAll);
  VecDestroy(gradTauNaturalAll);

  VecDestroy(sigNaturalAll);
  VecDestroy(gradSigNaturalAll);
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


void createSeqNodalImageVec(int Ne, int rank, int npes,
    const std::vector<double>& img, Vec & imgN0, MPI_Comm comm) {

  VecCreate(comm, &imgN0);

  if(!rank) {
    VecSetSizes(imgN0, (Ne + 1)*(Ne + 1)*(Ne + 1), PETSC_DECIDE);

    if(npes == 1) {
      VecSetType(imgN0, VECSEQ); 
    } else {
      VecSetType(imgN0, VECMPI); 
    }

    PetscScalar* imgArr;
    VecGetArray(imgN0, &imgArr);

    assert((img.size()) == (Ne*Ne*Ne));

    for(int k = 0 ; k < Ne; k++) {
      for(int j = 0 ; j < Ne; j++) {
        for(int i = 0 ; i < Ne; i++) {
          int ptIdxN = (((k*(Ne + 1)) + j)*(Ne + 1)) + i;
          int ptIdxE = (((k*Ne) + j)*Ne) + i;
          imgArr[ptIdxN] = img[ptIdxE];
        }
      }
    }

    for(int k = 0 ; k < Ne; k++) {
      for(int j = 0 ; j < Ne; j++) {
        imgArr[(((k*(Ne + 1)) + j)*(Ne + 1)) + Ne] = imgArr[(((k*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1];
      }
    }

    for(int k = 0 ; k < Ne; k++) {
      for(int i = 0 ; i < Ne; i++) {
        imgArr[(((k*(Ne + 1)) + Ne)*(Ne + 1)) + i] = imgArr[(((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i];
      }
    }

    for(int j = 0 ; j < Ne; j++) {
      for(int i = 0 ; i < Ne; i++) {
        imgArr[(((Ne*(Ne + 1)) + j)*(Ne + 1)) + i] = imgArr[((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + i];
      }
    }

    for(int i = 0 ; i < Ne; i++) {
      imgArr[(((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + i] = imgArr[((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i];
    }

    for(int j = 0 ; j < Ne; j++) {
      imgArr[(((Ne*(Ne + 1)) + j)*(Ne + 1)) + Ne] = imgArr[((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1];
    }

    for(int k = 0 ; k < Ne; k++) {
      imgArr[(((k*(Ne + 1)) + Ne)*(Ne + 1)) + Ne] = imgArr[(((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1];
    }

    imgArr[(((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + Ne] = imgArr[((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1];

    VecRestoreArray(imgN0, &imgArr);
  } else {
    VecSetSizes(imgN0, 0, PETSC_DECIDE);
    VecSetType(imgN0, VECMPI); 
  }//end if p0
}//end fn.


void coarsenImage(int Nfe, const std::vector<double>& imgF,
    std::vector<double>& imgC) {

  int Nce = Nfe/2;

  imgC.resize(Nce*Nce*Nce);

  assert(imgF.size() == (Nfe*Nfe*Nfe));

  for(int k = 0; k < Nce; k++) {
    for(int j = 0; j < Nce; j++) {
      for(int i = 0; i < Nce; i++) {
        assert( ((((((2*k) + 1)*Nfe) + ((2*j) + 1))*Nfe) + ((2*i) + 1)) < (Nfe*Nfe*Nfe) );
        imgC[(((k*Nce) + j)*Nce) + i] = 0.125*(
            imgF[((((2*k)*Nfe) + (2*j))*Nfe) + (2*i)] +
            imgF[((((2*k)*Nfe) + (2*j))*Nfe) + ((2*i) + 1)] +
            imgF[((((2*k)*Nfe) + ((2*j) + 1))*Nfe) + (2*i)] +
            imgF[((((2*k)*Nfe) + ((2*j) + 1))*Nfe) + ((2*i) + 1)] +
            imgF[(((((2*k) + 1)*Nfe) + (2*j))*Nfe) + (2*i)] +
            imgF[(((((2*k) + 1)*Nfe) + (2*j))*Nfe) + ((2*i) + 1)] +
            imgF[(((((2*k) + 1)*Nfe) + ((2*j) + 1))*Nfe) + (2*i)] +
            imgF[(((((2*k) + 1)*Nfe) + ((2*j) + 1))*Nfe) + ((2*i) + 1)]);
      }
    }
  }
}


void writeImage(char* fnamePrefix, int nx, int ny, int nz, std::vector<double> & img) {
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

  sprintf(fname, "%s.img", fnamePrefix);
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


void createCheckerBoardImage(int Ne, int width, Vec v) {
  //This is sequential
  VecZeroEntries(v);

  PetscScalar* arr;
  VecGetArray(v, &arr);

  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        int kOdd = ((k/width)%2);
        int jOdd = ((j/width)%2);
        int iOdd = ((i/width)%2);
        if((kOdd + jOdd + iOdd) == 2) {
          arr[(((k*Ne) + j)*Ne) + i] = 255.0;
        }
      }
    }
  }

  VecRestoreArray(v, &arr);
}


void createFineAnalyticFixedImage(int Ne, int width, Vec v) {
  //This is sequential
  VecZeroEntries(v);

  PetscScalar* arr;
  VecGetArray(v, &arr);

  double maxU, minU;
  double maxV, minV;

  double h = 1.0/static_cast<double>(Ne);
  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        double x0 = static_cast<double>(i)*h;
        double y = static_cast<double>(j)*h;
        double z = static_cast<double>(k)*h;

        double u1 = 0.05*sin(2.0*__PI__*x0*12.8)*
          sin(2.0*__PI__*y*12.8)*sin(2.0*__PI__*z*12.8);

        if( (i == 0) && (j == 0) && (k == 0) ) {
          maxU = u1;
          minU = u1;
        } else {
          if(u1 > maxU) {
            maxU  = u1;
          }
          if(u1 < minU) {
            minU = u1;
          }
        }

        double x = x0 + u1;

        double v = (sin(2.0*__PI__*x*6.4)*sin(2.0*__PI__*y*6.4)*sin(2.0*__PI__*z*6.4));

        if( (i == 0) && (j == 0) && (k == 0) ) {
          minV = v;
          maxV = v;
        } else {
          if(v < minV) {
            minV = v;
          }
          if(v > maxV) {
            maxV = v;
          }
        }

        arr[(((k*Ne) + j)*Ne) + i] = 255.0*v*v;
      }
    }
  }

  std::cout<<"Max U: "<<maxU<<" Min U: "<<minU<<std::endl;
  std::cout<<"Max I: "<<maxV<<" Min I: "<<minV<<std::endl;

  VecRestoreArray(v, &arr);
}


void createFineAnalyticImage(int Ne, int width, Vec v) {
  //This is sequential
  VecZeroEntries(v);

  PetscScalar* arr;
  VecGetArray(v, &arr);

  double maxV, minV;

  double h = 1.0/static_cast<double>(Ne);
  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        double x = static_cast<double>(i)*h;
        double y = static_cast<double>(j)*h;
        double z = static_cast<double>(k)*h;

        double v = (sin(2.0*__PI__*x*6.4)*sin(2.0*__PI__*y*6.4)*sin(2.0*__PI__*z*6.4));

        if( (i == 0) && (j == 0) && (k == 0) ) {
          minV = v;
          maxV = v;
        } else {
          if(v < minV) {
            minV = v;
          }
          if(v > maxV) {
            maxV = v;
          }
        }

        arr[(((k*Ne) + j)*Ne) + i] = 255.0*v*v;
      }
    }
  }

  std::cout<<"Max I: "<<maxV<<" Min I: "<<minV<<std::endl;

  VecRestoreArray(v, &arr);
}


void createAnalyticImage(int Ne, int width, Vec v) {
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
        double v = (sin(2.0*__PI__*x)*sin(2.0*__PI__*y)*sin(2.0*__PI__*z));
        arr[(((k*Ne) + j)*Ne) + i] = 255.0*v*v;
      }
    }
  }

  VecRestoreArray(v, &arr);
}


void createSphereImage(int Ne, int width, Vec v) {
  //This is sequential
  VecZeroEntries(v);

  PetscScalar* arr;
  VecGetArray(v, &arr);

  double radSqr = SQR(0.25);

  double h = 1.0/static_cast<double>(Ne);

  for(int k = 0; k < Ne; k++) {
    for(int j = 0; j < Ne; j++) {
      for(int i = 0; i < Ne; i++) {
        double x = static_cast<double>(i)*h;
        double y = static_cast<double>(j)*h;
        double z = static_cast<double>(k)*h;
        double distSqr = (SQR(x - 0.5)) + (SQR(y - 0.5)) + (SQR(z - 0.5)); 
        if(distSqr > radSqr) {
          //out
          arr[(((k*Ne) + j)*Ne) + i] = 0.0;
        } else  {
          //in
          arr[(((k*Ne) + j)*Ne) + i] = 255.0;
        }
      }//end for i
    }//end for j
  }//end for k

  VecRestoreArray(v, &arr);
}


void createCshapeImage(int Ne, int width, Vec v) {
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
        bool in = false;
        if( (SQR(0.25 - sqrt((SQR(x - 0.5)) + (SQR(y - 0.5))))) + (SQR(z - 0.5)) <= (SQR(0.1)) ) {
          if(x < 0.6) {
            in = true;
          }
        }
        if(in) {
          arr[(((k*Ne) + j)*Ne) + i] = 255.0;
        } else {
          arr[(((k*Ne) + j)*Ne) + i] = 0.0;
        }
      }//end for i
    }//end for j
  }//end for k

  VecRestoreArray(v, &arr);
}


void detJacMaxAndMin(DA da,  Vec u, double* maxDetJac,  double* minDetJac) {
  PetscInt Ne;
  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  PetscScalar**** uLocalArr;

  DAGetInfo(da, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL);

  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  //DA returns the number of nodes.
  //Need the number of elements.
  Ne--;

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


void newGaussNewton(ot::DAMG* damg, double fTol, double xTol,
    int maxIterCnt, double patchWidth, Vec Uin, Vec Uout) { 

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
  std::vector<ot::TreeNode>* imgPatches = ctxFinest->imgPatches;
  std::vector<unsigned int>* mesh = ctxFinest->mesh;
  std::vector<std::vector<double> >* tau = ctxFinest->tau;
  std::vector<std::vector<double> >* sigVals = ctxFinest->sigVals;
  std::vector<std::vector<double> >* gradTau = ctxFinest->gradTau;
  std::vector<std::vector<double> >* tauAtU = ctxFinest->tauAtU;
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
  updateNewHessContexts(damg, Uin);

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
  objValInit = evalObjective(daFinest, (*sigVals), (*tauAtU), numGpts, gWts,
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

      std::vector<std::vector<double> > tauAtUtmp;

      assert(imgPatches != NULL);
      assert(mesh != NULL);
      assert(tau != NULL);
      assert(gradTau != NULL);

      //8. Interpolate at Tmp Point
      computeNewTauAtU(daFinest, patchWidth, (*imgPatches),
          (*mesh), (*tau), (*gradTau), Uout, PhiMatStencil,
          numGpts, gPts, tauAtUtmp);

      assert(sigVals != NULL);

      //9.  Objective at tmp Point
      objVal = evalObjective(daFinest, (*sigVals), tauAtUtmp, numGpts, gWts,
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
    updateNewHessContexts(damg, Uin);

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

}


PetscErrorCode computeGaussianRHS(ot::DAMG damg, Vec rhs) {
  PetscFunctionBegin;

  ot::DA* da = damg->da;
  HessData* data = static_cast<HessData*>(damg->user);

  unsigned char* bdyArr = data->bdyArr;
  PetscScalar *inarray;

  PetscInt forceDir = 0;
  PetscOptionsGetInt(0, "-forceDir", &forceDir, 0);
  assert(forceDir < 3);
  assert(forceDir >= 0);

  VecZeroEntries(rhs);
  da->vecGetBuffer(rhs, inarray, false, false, false, 3);

  unsigned int maxD;
  unsigned int balOctmaxD;

  //Gaussian params
  double xc = 0.5;
  double yc = 0.5;
  double zc = 0.5;
  double sx = 0.05;
  double sy = 0.2;
  double sz = 0.4;

  int numGaussPts = 5;

  std::vector<double> wts(numGaussPts);
  std::vector<double> gPts(numGaussPts);

  //5-pt rule
  wts[0] = 0.568888889;  wts[1] = 0.47862867;  wts[2] =  0.47862867;
  wts[3] = 0.23692689; wts[4] = 0.23692689;
  gPts[0] = 0.0;  gPts[1] = 0.53846931; gPts[2] = -0.53846931;
  gPts[3] = 0.90617985; gPts[4] = -0.90617985;

  if(da->iAmActive()) {
    maxD = da->getMaxDepth();
    balOctmaxD = maxD - 1;
    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>())  
    {
      Point pt;
      pt = da->getCurrentOffset();
      unsigned int idx = da->curr();
      unsigned levelhere = (da->getLevel(idx) - 1);
      double hxOct = (double)((double)(1u << (balOctmaxD - levelhere))/(double)(1u << balOctmaxD));
      double x = (double)(pt.xint())/((double)(1u << (maxD - 1)));
      double y = (double)(pt.yint())/((double)(1u << (maxD - 1)));
      double z = (double)(pt.zint())/((double)(1u << (maxD - 1)));
      double fac = ((hxOct*hxOct*hxOct)/8.0);
      unsigned int indices[8];
      da->getNodeIndices(indices); 
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(unsigned int j = 0; j < 8; j++) {
          if(!bdyArr[indices[j]]) {
            double integral = 0.0;
            //Quadrature Rule
            for(int m = 0; m < numGaussPts; m++) {
              for(int n = 0; n < numGaussPts; n++) {
                for(int p = 0; p < numGaussPts; p++) {
                  double xPt = ( (hxOct*(1.0 +gPts[m])*0.5) + x );
                  double yPt = ( (hxOct*(1.0 + gPts[n])*0.5) + y );
                  double zPt = ( (hxOct*(1.0 + gPts[p])*0.5) + z );
                  double rhsVal = evalGuass3D(xc, yc, zc, sx, sy, sz, xPt, yPt, zPt); 
                  double ShFnVal = ( ot::ShapeFnCoeffs[childNum][elemType][j][0] + 
                      (ot::ShapeFnCoeffs[childNum][elemType][j][1]*gPts[m]) +
                      (ot::ShapeFnCoeffs[childNum][elemType][j][2]*gPts[n]) +
                      (ot::ShapeFnCoeffs[childNum][elemType][j][3]*gPts[p]) +
                      (ot::ShapeFnCoeffs[childNum][elemType][j][4]*gPts[m]*
                       gPts[n]) +
                      (ot::ShapeFnCoeffs[childNum][elemType][j][5]*gPts[n]*
                       gPts[p]) +
                      (ot::ShapeFnCoeffs[childNum][elemType][j][6]*gPts[p]*
                       gPts[m]) +
                      (ot::ShapeFnCoeffs[childNum][elemType][j][7]*gPts[m]*
                       gPts[n]*gPts[p]) );
                  integral += (wts[m]*wts[n]*wts[p]*rhsVal*ShFnVal);
                }
              }
            }
            inarray[(3*indices[j]) + forceDir] += (fac*integral);
          }//end if boundary
        }//end for j
    }//end ALL loop
  }//end if active

  da->vecRestoreBuffer(rhs, inarray, false, false, false, 3);

  VecScale(rhs, 10);

  PetscFunctionReturn(0);
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

  int rank = da->getRankAll();

  unsigned char* bdyArr = data->bdyArr;
  double mu = data->mu;
  double lambda = data->lambda;
  double alpha = data->alpha;
  int numGpts = data->numGpts;
  double* gWts = data->gWts;
  std::vector<std::vector<double> >* sigVals = data->sigVals;
  std::vector<std::vector<double> >* tauAtU = data->tauAtU; 
  std::vector<std::vector<double> >* gradTauAtU = data->gradTauAtU; 
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
      bdyArr, PhiMatStencil, numGpts, gWts, rhs);

  VecAXPY(rhs, -alpha, uTmp);

  enforceBC(da, bdyArr, rhs);

  PetscLogEventEnd(evalGradEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}


//Random Solution Consistent RHS
PetscErrorCode computeDummyRHS(ot::DAMG damg, Vec in) {
  PetscFunctionBegin;

  PetscTruth saveRandU;
  PetscOptionsHasName(0, "-saveRandU", &saveRandU);

  PetscTruth loadRandU;
  PetscOptionsHasName(0, "-loadRandU", &loadRandU);

  PetscTruth scatterRandU;
  PetscOptionsHasName(0, "-scatterRandU", &scatterRandU);

  int rank = damg->da->getRankAll();
  int npes = damg->da->getNpesAll();

  Vec U;
  VecDuplicate(in, &U);
  if(loadRandU) {
    char vecFname[256];
    sprintf(vecFname, "randR_%d_%d.dat",rank,npes);
    Vec Utmp;
    if(npes == 1) {
      loadSeqVector(Utmp, vecFname);
    } else {
      loadPrlVector(Utmp, vecFname);
    }
    if(scatterRandU) {
      PetscInt uSz;
      VecGetLocalSize(U, &uSz);
      int* scSendSz = new int[npes];
      assert(scSendSz);

      int* scSendOff = new int[npes];
      assert(scSendOff);

      int scRecvSz = uSz;

      par::Mpi_Gather<int>(&scRecvSz, scSendSz, 1, 0, MPI_COMM_WORLD);

      scSendOff[0] = 0;
      for(int i = 1; i < npes; i++) {
        scSendOff[i] = scSendOff[i - 1] + scSendSz[i - 1];
      }

      PetscScalar* tmpArr;
      PetscScalar* uArr;
      VecGetArray(Utmp, &tmpArr);
      VecGetArray(U, &uArr);

      MPI_Scatterv(tmpArr, scSendSz, scSendOff, MPI_DOUBLE,
          uArr, scRecvSz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      VecRestoreArray(Utmp, &tmpArr);
      VecRestoreArray(U, &uArr);

      delete [] scSendSz;
      scSendSz = NULL;

      delete [] scSendOff;
      scSendOff = NULL;
    } else {
      VecCopy(Utmp, U);
    }
    VecDestroy(Utmp);
  } else {
    PetscRandom rctx;  
    PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
    PetscRandomSetType(rctx, PETSCRAND48);
    PetscInt randomSeed = 12345;
    PetscRandomSetSeed(rctx, randomSeed);
    PetscRandomSeed(rctx);
    PetscRandomSetFromOptions(rctx);
    VecSetRandom(U, rctx);
    PetscRandomDestroy(rctx);
    if(saveRandU) {
      char vecFname[256];
      sprintf(vecFname, "randR_%d_%d.dat",rank,npes);
      saveVector(U, vecFname);
    }
  }

  PetscTruth skipRhsMatMult;
  PetscOptionsHasName(0, "-skipRhsMatMult", &skipRhsMatMult);

  if(skipRhsMatMult) {
    VecCopy(U, in);
  } else {
    HessData* ctx = static_cast<HessData*>(damg->user);
    assert(ctx != NULL);
    enforceBC(damg->da, ctx->bdyArr, U);
    MatMult(damg->J, U, in);
  }

  VecDestroy(U);

  PetscFunctionReturn(0);
}


double evalObjective(ot::DA* da, const std::vector<std::vector<double> >& sigVals,
    const std::vector<std::vector<double> >& tauAtU, int numGpts, double* gWts,
    unsigned char* bdyArr, double**** LaplacianStencil, 
    double**** GradDivStencil, double mu, double lambda, double alpha, Vec U, Vec tmp) {

  PetscLogEventBegin(evalObjEvent, 0, 0, 0, 0);

  elasMatVec(da, bdyArr, LaplacianStencil, GradDivStencil, mu, lambda, U, tmp);

  double objImgPart = computeObjImgPart(da, sigVals, tauAtU, numGpts, gWts); 

  PetscScalar objElasPart; 
  VecTDot(U, tmp, &objElasPart);

  PetscLogEventEnd(evalObjEvent, 0, 0, 0, 0);

  return (0.5*(objImgPart + (alpha*objElasPart)));
}


double computeObjImgPart(ot::DA* da, const std::vector<std::vector<double> >& sigVals,
    const std::vector<std::vector<double> >& tauAtU, int numGpts, double* gWts) {

  double objImgPartLocal = 0.0;

  int numImages = sigVals.size();
  assert(numImages);

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
              assert(i < sigVals.size());
              assert(i < tauAtU.size());
              assert( ptsCtr < (sigVals[i].size()) );
              assert( ptsCtr < (tauAtU[i].size()) );
              localVal += ((sigVals[i][ptsCtr] - tauAtU[i][ptsCtr])*
                  (sigVals[i][ptsCtr] - tauAtU[i][ptsCtr]));
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


void computeGradientImgPart(ot::DA* da, const std::vector<std::vector<double> >& sigVals,
    const std::vector<std::vector<double> >& tauAtU,
    const std::vector<std::vector<double> >& gradTauAtU,
    unsigned char* bdyArr, double****** PhiMatStencil,
    int numGpts, double* gWts, Vec g) {

  int numImages = sigVals.size();
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
                assert(i < sigVals.size());
                assert(i < tauAtU.size());
                assert(i < gradTauAtU.size());
                assert(ptIdx < sigVals[i].size());
                assert(ptIdx < tauAtU[i].size());
                assert( ((3*ptIdx) + 2) < gradTauAtU[i].size() );
                for(int dof = 0; dof < 3; dof++) {
                  localVal[dof] += ((sigVals[i][ptIdx] - tauAtU[i][ptIdx])*gradTauAtU[i][(3*ptIdx) + dof]);
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


void createNewHessContexts(ot::DAMG* damg, double patchWidth, std::vector<ot::TreeNode>& imgPatches,
    std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& sig, const std::vector<std::vector<double> >& gradSig,
    const std::vector<std::vector<double> >& tau, const std::vector<std::vector<double> >& gradTau,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts,
    double mu, double lambda, double alpha) {

  assert(damg != NULL);

  int nlevels = damg[0]->nlevels; 

  for(int i = 0; i < nlevels; i++) {
    HessData* ctx = new HessData;
    assert(ctx);
    ctx->gtVec = new std::vector<double>;
    ctx->imgPatches = NULL;
    ctx->mesh = NULL;
    ctx->tau = NULL;
    ctx->gradTau = NULL;
    ctx->sigVals = NULL;
    ctx->tauAtU = NULL;
    ctx->gradTauAtU = NULL;
    ctx->bdyArr = NULL;        
    ctx->Jmat_private = NULL;
    ctx->inTmp = NULL;
    ctx->outTmp = NULL;
    ctx->U = NULL;
    ctx->uTmp = NULL;
    ctx->patchWidth = patchWidth;
    ctx->mu = mu;
    ctx->lambda = lambda;
    ctx->alpha = alpha;
    ctx->numGpts = numGpts;
    ctx->gWts = gWts;
    ctx->gPts = gPts;
    ctx->PhiMatStencil = PhiMatStencil;
    ctx->LaplacianStencil = LaplacianStencil;
    ctx->GradDivStencil = GradDivStencil;

    //We don't need Ne. So setting to garbage to test for illegal use.
    ctx->Ne = 0;

    if(i == (nlevels - 1)) {
      //Finest level
      damg[i]->da->createVector(ctx->uTmp, false, false, 3);
      ctx->imgPatches = new std::vector<ot::TreeNode>;
      ctx->mesh = new std::vector<unsigned int>;
      ctx->sigVals = new std::vector<std::vector<double> >;
      ctx->tau = new std::vector<std::vector<double> >; 
      ctx->gradTau = new std::vector<std::vector<double> >; 
      ctx->tauAtU = new std::vector<std::vector<double> >; 
      ctx->gradTauAtU = new std::vector<std::vector<double> >; 
      computeNewSigVals(damg[i]->da, patchWidth, imgPatches, mesh,
          sig, gradSig, numGpts, gPts, (*(ctx->sigVals)));
      //We can avoid this copy, but then we must trust the user to not change
      //tau and gradTau for the entire run.
      (*(ctx->tau)) = tau;
      (*(ctx->gradTau)) = gradTau;
      (*(ctx->imgPatches)) = imgPatches;
      (*(ctx->mesh)) = mesh;
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
}


void createHessContexts(ot::DAMG* damg, int Ne,
    const std::vector<std::vector<double> >& sig, const std::vector<std::vector<double> >& gradSig,
    const std::vector<std::vector<double> >& tau, const std::vector<std::vector<double> >& gradTau,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts,
    double mu, double lambda, double alpha) {

  PetscLogEventBegin(createHessContextEvent, 0, 0, 0, 0);

  assert(damg != NULL);

  int nlevels = damg[0]->nlevels; 

  for(int i = 0; i < nlevels; i++) {
    HessData* ctx = new HessData;
    assert(ctx);
    ctx->gtVec = new std::vector<double>;
    ctx->imgPatches = NULL;
    ctx->mesh = NULL;
    ctx->tau = NULL;
    ctx->gradTau = NULL;
    ctx->sigVals = NULL;
    ctx->tauAtU = NULL;
    ctx->gradTauAtU = NULL;
    ctx->bdyArr = NULL;        
    ctx->Jmat_private = NULL;
    ctx->inTmp = NULL;
    ctx->outTmp = NULL;
    ctx->U = NULL;
    ctx->uTmp = NULL;
    ctx->mu = mu;
    ctx->lambda = lambda;
    ctx->alpha = alpha;
    ctx->numGpts = numGpts;
    ctx->gWts = gWts;
    ctx->gPts = gPts;
    ctx->Ne = Ne;
    ctx->PhiMatStencil = PhiMatStencil;
    ctx->LaplacianStencil = LaplacianStencil;
    ctx->GradDivStencil = GradDivStencil;

    if(i == (nlevels - 1)) {
      //Finest level
      damg[i]->da->createVector(ctx->uTmp, false, false, 3);
      ctx->sigVals = new std::vector<std::vector<double> >;
      ctx->tau = new std::vector<std::vector<double> >; 
      ctx->gradTau = new std::vector<std::vector<double> >; 
      ctx->tauAtU = new std::vector<std::vector<double> >; 
      ctx->gradTauAtU = new std::vector<std::vector<double> >; 
      computeSigVals(damg[i]->da, Ne, sig, gradSig, numGpts, gPts, (*(ctx->sigVals)));
      //We can avoid this copy, but then we must trust the user to not change
      //tau and gradTau for the entire run.
      (*(ctx->tau)) = tau;
      (*(ctx->gradTau)) = gradTau;
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
      ((damg[i])->da_aux)->vecGetBuffer<unsigned char>(tmpBdyFlagsAux, bdyArrAux,
                                                       false, false, true, 1);
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
          }
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
    if(ctx->imgPatches) {
      delete (ctx->imgPatches);
      ctx->imgPatches = NULL;
    }
    if(ctx->mesh) {
      delete (ctx->mesh);
      ctx->mesh = NULL;
    }
    if(ctx->tau) {
      delete (ctx->tau);
      ctx->tau = NULL;  
    }
    if(ctx->gradTau) {
      delete (ctx->gradTau);
      ctx->gradTau = NULL;  
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


PetscErrorCode updateNewHessContexts(ot::DAMG* damg, Vec U) {

  PetscFunctionBegin;

  PetscLogEventBegin(updateHessContextEvent, U, 0, 0, 0);

  assert(damg != NULL);

  int nlevels = damg[0]->nlevels;
  HessData* ctxFinest = static_cast<HessData*>(damg[nlevels - 1]->user);
  ot::DA* daFinest = damg[nlevels - 1]->da;

  assert(ctxFinest != NULL);
  assert(daFinest != NULL);

  ctxFinest->U = U;
  double****** PhiMatStencil = ctxFinest->PhiMatStencil;
  double patchWidth = ctxFinest->patchWidth;
  int numGpts = ctxFinest->numGpts;
  double* gPts = ctxFinest->gPts;
  std::vector<std::vector<double> >* tau = ctxFinest->tau;
  std::vector<std::vector<double> >* gradTau = ctxFinest->gradTau;
  std::vector<std::vector<double> >* tauAtU = ctxFinest->tauAtU;
  std::vector<std::vector<double> >* gradTauAtU = ctxFinest->gradTauAtU;
  std::vector<ot::TreeNode>* imgPatches = ctxFinest->imgPatches;
  std::vector<unsigned int>* mesh = ctxFinest->mesh;

  int rank = daFinest->getRankAll();

  assert(imgPatches != NULL);
  assert(mesh != NULL);
  assert(tau != NULL);
  assert(gradTau != NULL);
  assert(tauAtU != NULL);
  assert(gradTauAtU != NULL);

  //Compute the finest vector first
  std::vector<std::vector<double> > nodalGradTauAtU;
  computeNewNodalGradTauAtU(daFinest, patchWidth, (*imgPatches), (*mesh),
      (*tau), (*gradTau), U, nodalGradTauAtU);
  int numImages = nodalGradTauAtU.size();
  assert(numImages);

  computeNewTauAtU(daFinest, patchWidth, (*imgPatches), (*mesh), (*tau), (*gradTau), U, PhiMatStencil,
      numGpts, gPts, (*tauAtU));

  computeNewGradTauAtU(daFinest, patchWidth, (*imgPatches), (*mesh), (*tau), (*gradTau), U, PhiMatStencil,
      numGpts, gPts, (*gradTauAtU)); 

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
        assert( ((6*i) + (sixMap[dof1][dof2])) < (gtVecFinest->size()) );
        for(int j = 0; j < numImages; j++) {
          assert( ((3*i) + dof1) < nodalGradTauAtU[j].size() );
          assert( ((3*i) + dof2) < nodalGradTauAtU[j].size() );
          (*gtVecFinest)[(6*i) + (sixMap[dof1][dof2])] += 
            (nodalGradTauAtU[j][(3*i) + dof1]*
             nodalGradTauAtU[j][(3*i) + dof2]);
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

  PetscFunctionReturn(0);
}


PetscErrorCode updateHessContexts(ot::DAMG* damg, Vec U) {

  PetscFunctionBegin;

  PetscLogEventBegin(updateHessContextEvent, U, 0, 0, 0);

  int nlevels = damg[0]->nlevels;
  HessData* ctxFinest = static_cast<HessData*>(damg[nlevels - 1]->user);
  ot::DA* daFinest = damg[nlevels - 1]->da;

  ctxFinest->U = U;
  double****** PhiMatStencil = ctxFinest->PhiMatStencil;
  int numGpts = ctxFinest->numGpts;
  double* gPts = ctxFinest->gPts;
  int Ne = ctxFinest->Ne;
  std::vector<std::vector<double> >* tau = ctxFinest->tau;
  std::vector<std::vector<double> >* gradTau = ctxFinest->gradTau;
  std::vector<std::vector<double> >* tauAtU = ctxFinest->tauAtU;
  std::vector<std::vector<double> >* gradTauAtU = ctxFinest->gradTauAtU;

  int rank = daFinest->getRankAll();

  //Compute the finest vector first
  std::vector<std::vector<double> > nodalGradTauAtU;
  computeNodalGradTauAtU(daFinest, Ne, (*tau), (*gradTau), U, nodalGradTauAtU);
  int numImages = nodalGradTauAtU.size();

  computeTauAtU(daFinest, Ne, (*tau), (*gradTau), U, PhiMatStencil,
      numGpts, gPts, (*tauAtU));

  computeGradTauAtU(daFinest, Ne, (*tau), (*gradTau), U, PhiMatStencil,
      numGpts, gPts, (*gradTauAtU)); 

  std::vector<double>* gtVecFinest = ctxFinest->gtVec;
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
          (*gtVecFinest)[(6*i) + sixMap[dof1][dof2]] += 
            (nodalGradTauAtU[j][(3*i) + dof1]*
             nodalGradTauAtU[j][(3*i) + dof2]);
        }
      }
    }
  }

  //Coarsen using injection
  for(int i = (nlevels - 1); i > 0; i--) {
    HessData* ctxF = (static_cast<HessData*>(damg[i]->user));    
    HessData* ctxC = (static_cast<HessData*>(damg[i - 1]->user));
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

  PetscFunctionReturn(0);
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


void newComposeXposAtUnodal(ot::DA* dao, double patchWidth, Vec U, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr, 
    std::vector<unsigned int>& outOfBndsList) {

  xPosArr.clear();
  yPosArr.clear();
  zPosArr.clear();
  outOfBndsList.clear();

  if(dao->iAmActive()) {
    unsigned int maxD = dao->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);

    std::vector<double> xyzVec;
    std::vector<char> markedNode;

    dao->createVector<double>(xyzVec, false, false, 3);
    dao->createVector<char>(markedNode, false, false, 1);

    for(int i = 0; i < xyzVec.size(); i++) {
      xyzVec[i] = 0;
    }

    for(int i = 0; i < markedNode.size(); i++) {
      markedNode[i] = 0;
    }

    double* xyzArr = NULL;
    char* markedNodeArr = NULL;
    dao->vecGetBuffer<double>(xyzVec, xyzArr, false, false, false, 3);
    dao->vecGetBuffer<char>(markedNode, markedNodeArr, false, false, false, 1);

    PetscScalar* uArr;
    dao->vecGetBuffer(U, uArr, false, false, true, 3);

    unsigned int posBdyNodeSize = dao->getBoundaryNodeSize();

    //No communication. Each processor sets the values for the nodes it owns.

    if(posBdyNodeSize) {
      //This processor owns some positive boundary nodes.
      //We may need to loop over pre-ghosts to access them, for example if the
      //processor only owns positive boundary nodes and no elements.
      unsigned int myBegin = dao->getIdxElementBegin();
      unsigned int myEnd = dao->getIdxPostGhostBegin();
      for(dao->init<ot::DA_FLAGS::ALL>();
          dao->curr() < dao->end<ot::DA_FLAGS::ALL>();
          dao->next<ot::DA_FLAGS::ALL>()) {
        Point pt = dao->getCurrentOffset();
        unsigned int idx = dao->curr();
        unsigned int lev = dao->getLevel(idx);
        unsigned int xint = pt.xint();
        unsigned int yint = pt.yint();
        unsigned int zint = pt.zint();
        double hOct = hFac*(static_cast<double>(1u << (maxD - lev)));
        double x0 = hFac*static_cast<double>(xint);
        double y0 = hFac*static_cast<double>(yint);
        double z0 = hFac*static_cast<double>(zint);
        unsigned int indices[8];
        unsigned char hnMask = dao->getHangingNodeIndex(idx);
        unsigned char currentFlags;
        bool isBoundary = dao->isBoundaryOctant(&currentFlags);
        if(currentFlags > ot::TreeNode::NEG_POS_DEMARCATION) {
          //touches atleast one positive boundary
          dao->getNodeIndices(indices);
        } else {
          if(dao->isLUTcompressed()) {
            dao->updateQuotientCounter();
          }
        }
        int xPosBdy = (currentFlags & ot::TreeNode::X_POS_BDY);
        int yPosBdy = (currentFlags & ot::TreeNode::Y_POS_BDY);
        int zPosBdy = (currentFlags & ot::TreeNode::Z_POS_BDY);
        if(xPosBdy) {
          if( (indices[1] >= myBegin) && (indices[1] < myEnd) ) {
            if(!(hnMask & (1<<1))) {
              xyzArr[(3*indices[1])] = x0 + hOct + uArr[(3*indices[1])];
              xyzArr[(3*indices[1]) + 1] = y0 + uArr[(3*indices[1]) + 1];
              xyzArr[(3*indices[1]) + 2] = z0 + uArr[(3*indices[1]) + 2];
              if( (fabs(uArr[3*indices[1]]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[1]) + 1]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[1]) + 2]) >= patchWidth) ) {
                markedNodeArr[indices[1]] = 1;
              }
            }
          }
        }
        if(yPosBdy) {
          if( (indices[2] >= myBegin) && (indices[2] < myEnd) ) {
            if(!(hnMask & (1<<2))) {
              xyzArr[(3*indices[2])] = x0 + uArr[(3*indices[2])];
              xyzArr[(3*indices[2]) + 1] = y0 + hOct + uArr[(3*indices[2]) + 1];
              xyzArr[(3*indices[2]) + 2] = z0 + uArr[(3*indices[2]) + 2];
              if( (fabs(uArr[3*indices[2]]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[2]) + 1]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[2]) + 2]) >= patchWidth) ) {
                markedNodeArr[indices[2]] = 1;
              }
            }
          }
        }
        if(zPosBdy) {
          if( (indices[4] >= myBegin) && (indices[4] < myEnd) ) {
            if(!(hnMask & (1<<4))) {
              xyzArr[(3*indices[4])] = x0 + uArr[(3*indices[4])];
              xyzArr[(3*indices[4]) + 1] = y0 + uArr[(3*indices[4]) + 1];
              xyzArr[(3*indices[4]) + 2] = z0 + hOct + uArr[(3*indices[4]) + 2];
              if( (fabs(uArr[3*indices[4]]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[4]) + 1]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[4]) + 2]) >= patchWidth) ) {
                markedNodeArr[indices[4]] = 1;
              }
            }
          }
        }
        if(xPosBdy && yPosBdy) {
          if( (indices[3] >= myBegin) && (indices[3] < myEnd) ) {
            if(!(hnMask & (1<<3))) {
              xyzArr[(3*indices[3])] = x0 + hOct + uArr[(3*indices[3])];
              xyzArr[(3*indices[3]) + 1] = y0 + hOct + uArr[(3*indices[3]) + 1];
              xyzArr[(3*indices[3]) + 2] = z0 + uArr[(3*indices[3]) + 2];
              if( (fabs(uArr[3*indices[3]]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[3]) + 1]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[3]) + 2]) >= patchWidth) ) {
                markedNodeArr[indices[3]] = 1;
              }
            }
          }
        }
        if(xPosBdy && zPosBdy) {
          if( (indices[5] >= myBegin) && (indices[5] < myEnd) ) {
            if(!(hnMask & (1<<5))) {
              xyzArr[(3*indices[5])] = x0 + hOct + uArr[(3*indices[5])];
              xyzArr[(3*indices[5]) + 1] = y0 + uArr[(3*indices[5]) + 1];
              xyzArr[(3*indices[5]) + 2] = z0 + hOct + uArr[(3*indices[5]) + 2];
              if( (fabs(uArr[3*indices[5]]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[5]) + 1]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[5]) + 2]) >= patchWidth) ) {
                markedNodeArr[indices[5]] = 1;
              }
            }
          }
        }
        if(yPosBdy && zPosBdy) {
          if( (indices[6] >= myBegin) && (indices[6] < myEnd) ) {
            if(!(hnMask & (1<<6))) {
              xyzArr[(3*indices[6])] = x0 + uArr[(3*indices[6])];
              xyzArr[(3*indices[6]) + 1] = y0 + hOct + uArr[(3*indices[6]) + 1];
              xyzArr[(3*indices[6]) + 2] = z0 + hOct + uArr[(3*indices[6]) + 2];
              if( (fabs(uArr[3*indices[6]]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[6]) + 1]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[6]) + 2]) >= patchWidth) ) {
                markedNodeArr[indices[6]] = 1;
              }
            }
          }
        }
        if(xPosBdy && yPosBdy && zPosBdy) {
          if( (indices[7] >= myBegin) && (indices[7] < myEnd) ) {
            if(!(hnMask & (1<<7))) {
              xyzArr[(3*indices[7])] = x0 + hOct + uArr[(3*indices[7])];
              xyzArr[(3*indices[7]) + 1] = y0 + hOct + uArr[(3*indices[7]) + 1];
              xyzArr[(3*indices[7]) + 2] = z0 + hOct + uArr[(3*indices[7]) + 2];
              if( (fabs(uArr[3*indices[7]]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[7]) + 1]) >= patchWidth) ||
                  (fabs(uArr[(3*indices[7]) + 2]) >= patchWidth) ) {
                markedNodeArr[indices[7]] = 1;
              }
            }
          }
        }
        if(idx >= myBegin) {
          if(!(hnMask & 1)) {
            xyzArr[(3*idx)] = x0 + uArr[(3*idx)];
            xyzArr[(3*idx) + 1] = y0 + uArr[(3*idx) + 1];
            xyzArr[(3*idx) + 2] = z0 + uArr[(3*idx) + 2];
            if( (fabs(uArr[3*idx]) >= patchWidth) ||
                (fabs(uArr[(3*idx) + 1]) >= patchWidth) ||
                (fabs(uArr[(3*idx) + 2]) >= patchWidth) ) {
              markedNodeArr[idx] = 1;
            }
          }
        }
      }//end ALL
    } else {
      //This processor does not own any positive boundary nodes.
      //Only need to write to anchors.
      for(dao->init<ot::DA_FLAGS::WRITABLE>();
          dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
          dao->next<ot::DA_FLAGS::WRITABLE>()) {
        Point pt = dao->getCurrentOffset();
        unsigned int idx = dao->curr();
        unsigned int xint = pt.xint();
        unsigned int yint = pt.yint();
        unsigned int zint = pt.zint();
        double x0 = hFac*static_cast<double>(xint);
        double y0 = hFac*static_cast<double>(yint);
        double z0 = hFac*static_cast<double>(zint);
        unsigned char hnMask = dao->getHangingNodeIndex(idx);
        if(!(hnMask & 1)) {
          //Anchor is not hanging
          xyzArr[(3*idx)] = x0 + uArr[(3*idx)];
          xyzArr[(3*idx) + 1] = y0 + uArr[(3*idx) + 1];
          xyzArr[(3*idx) + 2] = z0 + uArr[(3*idx) + 2];
          if( (fabs(uArr[3*idx]) >= patchWidth) ||
              (fabs(uArr[(3*idx) + 1]) >= patchWidth) ||
              (fabs(uArr[(3*idx) + 2]) >= patchWidth) ) {
            markedNodeArr[idx] = 1;
          }
        }
      }//end WRITABLE
    }

    dao->vecRestoreBuffer(U, uArr, false, false, true, 3);

    dao->vecRestoreBuffer<double>(xyzVec, xyzArr, false, false, false, 3);
    dao->vecRestoreBuffer<char>(markedNode, markedNodeArr, false, false, false, 1);

    unsigned int numPts = (xyzVec.size())/3;
    xPosArr.resize(numPts);
    yPosArr.resize(numPts);
    zPosArr.resize(numPts);
    for(int i = 0; i < numPts; i++) {
      xPosArr[i] = xyzVec[3*i];
      yPosArr[i] = xyzVec[(3*i) + 1];
      zPosArr[i] = xyzVec[(3*i) + 2];
      if(markedNode[i]) {
        outOfBndsList.push_back(i);
      }
    }//end for i
  }//end if active
}


void composeXposAtUnodal(ot::DA* dao, Vec U, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr) {

  xPosArr.clear();
  yPosArr.clear();
  zPosArr.clear();

  if(dao->iAmActive()) {
    unsigned int maxD = dao->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);

    std::vector<double> xyzVec;

    dao->createVector<double>(xyzVec, false, false, 3);

    for(int i = 0; i < xyzVec.size(); i++) {
      xyzVec[i] = 0;
    }

    double* xyzArr = NULL;
    dao->vecGetBuffer<double>(xyzVec, xyzArr, false, false, false, 3);

    PetscScalar* uArr;
    dao->vecGetBuffer(U, uArr, false, false, true, 3);

    unsigned int posBdyNodeSize = dao->getBoundaryNodeSize();

    //No communication. Each processor sets the values for the nodes it owns.

    if(posBdyNodeSize) {
      //This processor owns some positive boundary nodes.
      //We may need to loop over pre-ghosts to access them, for example if the
      //processor only owns positive boundary nodes and no elements.
      unsigned int myBegin = dao->getIdxElementBegin();
      unsigned int myEnd = dao->getIdxPostGhostBegin();
      for(dao->init<ot::DA_FLAGS::ALL>();
          dao->curr() < dao->end<ot::DA_FLAGS::ALL>();
          dao->next<ot::DA_FLAGS::ALL>()) {
        Point pt = dao->getCurrentOffset();
        unsigned int idx = dao->curr();
        unsigned int lev = dao->getLevel(idx);
        unsigned int xint = pt.xint();
        unsigned int yint = pt.yint();
        unsigned int zint = pt.zint();
        double hOct = hFac*(static_cast<double>(1u << (maxD - lev)));
        double x0 = hFac*static_cast<double>(xint);
        double y0 = hFac*static_cast<double>(yint);
        double z0 = hFac*static_cast<double>(zint);
        unsigned int indices[8];
        unsigned char hnMask = dao->getHangingNodeIndex(idx);
        unsigned char currentFlags;
        bool isBoundary = dao->isBoundaryOctant(&currentFlags);
        if(currentFlags > ot::TreeNode::NEG_POS_DEMARCATION) {
          //touches atleast one positive boundary
          dao->getNodeIndices(indices);
        } else {
          if(dao->isLUTcompressed()) {
            dao->updateQuotientCounter();
          }
        }
        int xPosBdy = (currentFlags & ot::TreeNode::X_POS_BDY);
        int yPosBdy = (currentFlags & ot::TreeNode::Y_POS_BDY);
        int zPosBdy = (currentFlags & ot::TreeNode::Z_POS_BDY);
        if(xPosBdy) {
          if( (indices[1] >= myBegin) && (indices[1] < myEnd) ) {
            if(!(hnMask & (1<<1))) {
              xyzArr[(3*indices[1])] = x0 + hOct + uArr[(3*indices[1])];
              xyzArr[(3*indices[1]) + 1] = y0 + uArr[(3*indices[1]) + 1];
              xyzArr[(3*indices[1]) + 2] = z0 + uArr[(3*indices[1]) + 2];
            }
          }
        }
        if(yPosBdy) {
          if( (indices[2] >= myBegin) && (indices[2] < myEnd) ) {
            if(!(hnMask & (1<<2))) {
              xyzArr[(3*indices[2])] = x0 + uArr[(3*indices[2])];
              xyzArr[(3*indices[2]) + 1] = y0 + hOct + uArr[(3*indices[2]) + 1];
              xyzArr[(3*indices[2]) + 2] = z0 + uArr[(3*indices[2]) + 2];
            }
          }
        }
        if(zPosBdy) {
          if( (indices[4] >= myBegin) && (indices[4] < myEnd) ) {
            if(!(hnMask & (1<<4))) {
              xyzArr[(3*indices[4])] = x0 + uArr[(3*indices[4])];
              xyzArr[(3*indices[4]) + 1] = y0 + uArr[(3*indices[4]) + 1];
              xyzArr[(3*indices[4]) + 2] = z0 + hOct + uArr[(3*indices[4]) + 2];
            }
          }
        }
        if(xPosBdy && yPosBdy) {
          if( (indices[3] >= myBegin) && (indices[3] < myEnd) ) {
            if(!(hnMask & (1<<3))) {
              xyzArr[(3*indices[3])] = x0 + hOct + uArr[(3*indices[3])];
              xyzArr[(3*indices[3]) + 1] = y0 + hOct + uArr[(3*indices[3]) + 1];
              xyzArr[(3*indices[3]) + 2] = z0 + uArr[(3*indices[3]) + 2];
            }
          }
        }
        if(xPosBdy && zPosBdy) {
          if( (indices[5] >= myBegin) && (indices[5] < myEnd) ) {
            if(!(hnMask & (1<<5))) {
              xyzArr[(3*indices[5])] = x0 + hOct + uArr[(3*indices[5])];
              xyzArr[(3*indices[5]) + 1] = y0 + uArr[(3*indices[5]) + 1];
              xyzArr[(3*indices[5]) + 2] = z0 + hOct + uArr[(3*indices[5]) + 2];
            }
          }
        }
        if(yPosBdy && zPosBdy) {
          if( (indices[6] >= myBegin) && (indices[6] < myEnd) ) {
            if(!(hnMask & (1<<6))) {
              xyzArr[(3*indices[6])] = x0 + uArr[(3*indices[6])];
              xyzArr[(3*indices[6]) + 1] = y0 + hOct + uArr[(3*indices[6]) + 1];
              xyzArr[(3*indices[6]) + 2] = z0 + hOct + uArr[(3*indices[6]) + 2];
            }
          }
        }
        if(xPosBdy && yPosBdy && zPosBdy) {
          if( (indices[7] >= myBegin) && (indices[7] < myEnd) ) {
            if(!(hnMask & (1<<7))) {
              xyzArr[(3*indices[7])] = x0 + hOct + uArr[(3*indices[7])];
              xyzArr[(3*indices[7]) + 1] = y0 + hOct + uArr[(3*indices[7]) + 1];
              xyzArr[(3*indices[7]) + 2] = z0 + hOct + uArr[(3*indices[7]) + 2];
            }
          }
        }
        if(idx >= myBegin) {
          if(!(hnMask & 1)) {
            xyzArr[(3*idx)] = x0 + uArr[(3*idx)];
            xyzArr[(3*idx) + 1] = y0 + uArr[(3*idx) + 1];
            xyzArr[(3*idx) + 2] = z0 + uArr[(3*idx) + 2];
          }
        }
      }//end ALL
    } else {
      //This processor does not own any positive boundary nodes.
      //Only need to write to anchors.
      for(dao->init<ot::DA_FLAGS::WRITABLE>();
          dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
          dao->next<ot::DA_FLAGS::WRITABLE>()) {
        Point pt = dao->getCurrentOffset();
        unsigned int idx = dao->curr();
        unsigned int xint = pt.xint();
        unsigned int yint = pt.yint();
        unsigned int zint = pt.zint();
        double x0 = hFac*static_cast<double>(xint);
        double y0 = hFac*static_cast<double>(yint);
        double z0 = hFac*static_cast<double>(zint);
        unsigned char hnMask = dao->getHangingNodeIndex(idx);
        if(!(hnMask & 1)) {
          //Anchor is not hanging
          xyzArr[(3*idx)] = x0 + uArr[(3*idx)];
          xyzArr[(3*idx) + 1] = y0 + uArr[(3*idx) + 1];
          xyzArr[(3*idx) + 2] = z0 + uArr[(3*idx) + 2];
        }
      }//end WRITABLE
    }

    dao->vecRestoreBuffer(U, uArr, false, false, true, 3);

    dao->vecRestoreBuffer<double>(xyzVec, xyzArr, false, false, false, 3);

    unsigned int numPts = (xyzVec.size())/3;
    xPosArr.resize(numPts);
    yPosArr.resize(numPts);
    zPosArr.resize(numPts);
    for(int i = 0; i < numPts; i++) {
      xPosArr[i] = xyzVec[3*i];
      yPosArr[i] = xyzVec[(3*i) + 1];
      zPosArr[i] = xyzVec[(3*i) + 2];
    }
  }//end if active
}


void composeXposAtU(ot::DA* dao, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr) {

  typedef double* doublePtr;
  typedef double** double2Ptr;

  xPosArr.clear();
  yPosArr.clear();
  zPosArr.clear();

  if(dao->iAmActive()) {
    unsigned int maxD = dao->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);

    double*** u1loc = new double2Ptr[numGpts];
    double*** u2loc = new double2Ptr[numGpts];
    double*** u3loc = new double2Ptr[numGpts];
    for(int m = 0; m < numGpts; m++){
      u1loc[m] = new doublePtr[numGpts];
      u2loc[m] = new doublePtr[numGpts];
      u3loc[m] = new doublePtr[numGpts];
      for(int n = 0; n < numGpts; n++){
        u1loc[m][n] = new double[numGpts];
        u2loc[m][n] = new double[numGpts];
        u3loc[m][n] = new double[numGpts];
      }
    }

    PetscScalar* uArr;
    dao->vecGetBuffer(U, uArr, false, false, true, 3);

    //We can overlap communication with computation (later) 
    //Note, if we overlap communication with computation then WRITABLE will be
    //split into two: W_DEP.. and INDEP..
    //The same loop must be used even in computing sigVals (although there is
    //no communication there) because the list of values here and there must be
    //"aligned"
    //Also, the same loop must be used when using these vectors, such as in the
    //objective function or gradient evaluation
    dao->ReadFromGhostsBegin<PetscScalar>(uArr, 3);
    dao->ReadFromGhostsEnd<PetscScalar>(uArr);

    for(dao->init<ot::DA_FLAGS::WRITABLE>();
        dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
        dao->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = dao->getCurrentOffset();
      unsigned int idx = dao->curr();
      unsigned int lev = dao->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      double hOct = hFac*(static_cast<double>(1u << (maxD - lev)));
      double x0 = hFac*static_cast<double>(xint);
      double y0 = hFac*static_cast<double>(yint);
      double z0 = hFac*static_cast<double>(zint);
      unsigned int indices[8];
      dao->getNodeIndices(indices);
      unsigned char childNum = dao->getChildNumber();
      unsigned char hnMask = dao->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType, hnMask, childNum);
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            u1loc[m][n][p] = 0;
            u2loc[m][n][p] = 0;
            u3loc[m][n][p] = 0;
            for(int k = 0; k < 8; k++) {
              double PhiMatVal = PhiMatStencil[childNum][elemType][k][m][n][p];
              u1loc[m][n][p] += (uArr[(3*indices[k])]*PhiMatVal);
              u2loc[m][n][p] += (uArr[(3*indices[k]) + 1]*PhiMatVal);
              u3loc[m][n][p] += (uArr[(3*indices[k]) + 2]*PhiMatVal);
            }
          }
        }
      }
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            xPosArr.push_back(x0 + (0.5*hOct*(1.0 + gPts[m])) + u1loc[m][n][p]);
            yPosArr.push_back(y0 + (0.5*hOct*(1.0 + gPts[n])) + u2loc[m][n][p]);
            zPosArr.push_back(z0 + (0.5*hOct*(1.0 + gPts[p])) + u3loc[m][n][p]);
          }
        }
      }
    }

    dao->vecRestoreBuffer(U, uArr, false, false, true, 3);

    for(int m = 0; m < numGpts; m++){
      for(int n = 0; n < numGpts; n++){
        delete [] u1loc[m][n];
        u1loc[m][n] = NULL;

        delete [] u2loc[m][n];
        u2loc[m][n] = NULL;

        delete [] u3loc[m][n];
        u3loc[m][n] = NULL;
      }
      delete [] u1loc[m];
      u1loc[m] = NULL;

      delete [] u2loc[m];
      u2loc[m] = NULL;

      delete [] u3loc[m];
      u3loc[m] = NULL;
    }
    delete [] u1loc;
    u1loc = NULL;

    delete [] u2loc;
    u2loc = NULL;

    delete [] u3loc;
    u3loc = NULL;
  }
}


void computeFDgradient(DA dai, DA dao, Vec in, Vec out) {
  PetscInt Ne;
  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  PetscScalar*** iArr;
  PetscScalar**** oArr;
  Vec iLoc;

  DAGetLocalVector(dai, &iLoc);
  DAGlobalToLocalBegin(dai, in, INSERT_VALUES, iLoc);

  DAGetInfo(dai, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL);
  DAGetCorners(dai, &xs, &ys, &zs, &nx, &ny, &nz);

  //DA returns the number of nodes.
  //Need the number of elements.
  Ne--;

  double h = 1.0/static_cast<double>(Ne);

  VecZeroEntries(out);
  DAVecGetArrayDOF(dao, out, &oArr);

  DAGlobalToLocalEnd(dai, in, INSERT_VALUES, iLoc);

  DAVecGetArray(dai, iLoc, &iArr);
  for(int k = zs; k < zs + nz; k++) {
    for(int j = ys; j < ys + ny; j++) {
      for(int i = xs; i < xs + nx; i++) {
        PetscScalar fxplus = 0;
        PetscScalar fyplus = 0;
        PetscScalar fzplus = 0;
        PetscScalar fxminus = 0;
        PetscScalar fyminus = 0;
        PetscScalar fzminus = 0;
        //Only need to take care of domain boundaries
        //Processor boundaries are okay, since we work with ghosted (local) vectors
        if(i) {
          fxminus = iArr[k][j][i - 1];
        }
        if(i < Ne) {
          fxplus = iArr[k][j][i + 1];
        }
        if(j) {
          fyminus = iArr[k][j - 1][i];
        }
        if(j < Ne) {
          fyplus = iArr[k][j + 1][i];
        }
        if(k) {
          fzminus = iArr[k - 1][j][i];
        }
        if(k < Ne) {
          fzplus = iArr[k + 1][j][i];
        }
        oArr[k][j][i][0] = (fxplus - fxminus)/(2.0*h);
        oArr[k][j][i][1] = (fyplus - fyminus)/(2.0*h);
        oArr[k][j][i][2] = (fzplus - fzminus)/(2.0*h);
      }
    }
  }

  DAVecRestoreArray(dai, iLoc, &iArr);
  DARestoreLocalVector(dai, &iLoc);
  DAVecRestoreArrayDOF(dao, out, &oArr);
}


void computeFDhessian(DA dai, DA dao, Vec in, Vec out) {
  PetscInt Ne;
  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  PetscScalar*** iArr;
  PetscScalar**** oArr;

  Vec iLoc;
  DAGetLocalVector(dai, &iLoc);

  DAGlobalToLocalBegin(dai, in, INSERT_VALUES, iLoc);

  DAGetInfo(dai, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL);

  DAGetCorners(dai, &xs, &ys, &zs, &nx, &ny, &nz);

  //DA returns the number of nodes.
  //Need the number of elements.
  Ne--;

  double h = 1.0/static_cast<double>(Ne);
  double hsq = h*h;

  VecZeroEntries(out);

  DAVecGetArrayDOF(dao, out, &oArr);

  DAGlobalToLocalEnd(dai, in, INSERT_VALUES, iLoc);

  DAVecGetArray(dai, iLoc, &iArr);
  for(int k = zs; k < zs + nz; k++) {
    for(int j = ys; j < ys + ny; j++) {
      for(int i = xs; i < xs + nx; i++) {
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
          fxminus = iArr[k][j][i - 1];
          if(j < Ne) {
            fxmyp = iArr[k][j + 1][i - 1];
          }
          if(j) {
            fxmym = iArr[k][j - 1][i - 1];
          }
          if(k < Ne) {
            fxmzp = iArr[k + 1][j][i - 1];
          }
          if(k) {
            fxmzm = iArr[k - 1][j][i - 1];
          }
        }
        if(i < Ne) {
          fxplus = iArr[k][j][i + 1];
          if(j < Ne) {
            fxpyp = iArr[k][j + 1][i + 1];
          }
          if (j) {
            fxpym = iArr[k][j - 1][i + 1];
          }
          if (k < Ne) {
            fxpzp = iArr[k + 1][j][i + 1];
          }
          if (k) {
            fxpzm = iArr[k - 1][j][i + 1];
          }
        }
        if(j) {
          fyminus = iArr[k][j - 1][i];
          if(k < Ne) {
            fymzp = iArr[k + 1][j - 1][i];
          }
          if(k) {
            fymzm = iArr[k - 1][j - 1][i];
          }
        }
        if(j < Ne) {
          fyplus = iArr[k][j + 1][i];
          if(k < Ne) {
            fypzp = iArr[k + 1][j + 1][i];
          }
          if(k) {
            fypzm = iArr[k - 1][j + 1][i];
          } 
        }
        if(k) {
          fzminus = iArr[k - 1][j][i];
        }
        if(k < Ne) {
          fzplus = iArr[k + 1][j][i];
        }
        //h12
        oArr[k][j][i][0] = (fxpyp + fxmym - (fxpym + fxmyp))/(4.0*hsq);

        //h13
        oArr[k][j][i][1] =  (fxpzp + fxmzm - (fxpzm + fxmzp))/(4.0*hsq);

        //h23
        oArr[k][j][i][2] =  (fypzp + fymzm - (fypzm + fymzp))/(4.0*hsq);

        //h11
        oArr[k][j][i][3] =  (fxplus + fxminus - (2.0*iArr[k][j][i]))/hsq;

        //h22
        oArr[k][j][i][4] =  (fyplus + fyminus - (2.0*iArr[k][j][i]))/hsq;

        //h33
        oArr[k][j][i][5] =  (fzplus + fzminus - (2.0*iArr[k][j][i]))/hsq;
      }
    }
  }

  DAVecRestoreArray(dai, iLoc, &iArr);
  DAVecRestoreArrayDOF(dao, out, &oArr);
  DARestoreLocalVector(dai, &iLoc);
}


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


double evalGuass3D(double x0, double y0, double z0,
    double sx, double sy, double sz, double x, double y, double z)  {

  double expArg = -( ((SQR(x - x0))/(2.0*(SQR(sx)))) + 
      ((SQR(y - y0))/(2.0*(SQR(sy)))) +
      ((SQR(z - z0))/(2.0*(SQR(sz)))) );

  double fVal = exp(expArg);

  return fVal;
}




