
/**
  @file registration.C
  @brief Components of Elastic Registration
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  */

#include "mpi.h"
#include "oda.h"
#include "petscda.h"
#include "seqUtils.h"
#include <cmath>
#include <iostream>
#include "registration.h"
#include "regInterpQuintic.h"

#define CUBE(a) ((a)*(a)*(a))

#define SQR(a) ((a)*(a))

void computeQuinticGradTauAtU(ot::DA* dao, DA dar1, DA dar2, DA dar3,
    Vec tauLoc, Vec gradTauLoc, Vec hessTauLoc, Vec U,
    const double****** PhiMatStencil, int numGpts, const double* gPts, std::vector<double>& gradTauAtU) { 

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  composeXposAtU(dao, U, PhiMatStencil, numGpts, gPts, xPosArr, yPosArr, zPosArr);

  MPI_Comm commAll = dao->getComm();

  evalQuinticGradFnAtAllPts(xPosArr, yPosArr, zPosArr, dar1, dar2, dar3,
      tauLoc, gradTauLoc, hessTauLoc, gradTauAtU, commAll);

}

//PETSC DA uses all the processors, Octree DA may only use a subset of this
//U is NOT a local vector, i.e. ghosts are not duplicated
//U is assumed to come directly from the solver
//tauLoc, gradTauLoc and hessTauLoc are local vectors, so they already have the
//ghost values.
void computeQuinticTauAtU(ot::DA* dao, DA dar1, DA dar2, DA dar3,
    Vec tauLoc, Vec gradTauLoc, Vec hessTauLoc, Vec U,
    const double****** PhiMatStencil, int numGpts, const double* gPts, std::vector<double>& tauAtU) {

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  composeXposAtU(dao, U, PhiMatStencil, numGpts, gPts, xPosArr, yPosArr, zPosArr);

  MPI_Comm commAll = dao->getComm();

  evalQuinticFnAtAllPts(xPosArr, yPosArr, zPosArr, dar1, dar2, dar3,
      tauLoc, gradTauLoc, hessTauLoc, tauAtU, commAll);

}

//PETSC DA uses all the processors, Octree DA may only use a subset of this
//sigLoc, gradSigLoc and hessSigLoc are local vectors, so they already have the
//ghost values.
void computeQuinticSigVals(ot::DA* dao, DA dar1, DA dar2, DA dar3,
    Vec sigLoc, Vec gradSigLoc, Vec hessSigLoc,
    int numGpts, const double* gPts, std::vector<double>& sigVals) {

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;
  if(dao->iAmActive()) {
    unsigned int maxD = dao->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);
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
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            xPosArr.push_back(x0 + (0.5*hOct*(1.0 + gPts[m])));
            yPosArr.push_back(y0 + (0.5*hOct*(1.0 + gPts[n])));
            zPosArr.push_back(z0 + (0.5*hOct*(1.0 + gPts[p])));
          }
        }
      }
    }
  }

  MPI_Comm commAll = dao->getComm();

  evalQuinticFnAtAllPts(xPosArr, yPosArr, zPosArr, dar1, dar2, dar3,
      sigLoc, gradSigLoc, hessSigLoc, sigVals, commAll);
}

void evalQuinticGradFnAtAllPts(std::vector<double>& xPosArr, std::vector<double>& yPosArr,
    std::vector<double>& zPosArr, DA dar1, DA dar2, DA dar3,
    Vec fnLoc, Vec gradLoc, Vec hessLoc, std::vector<double>& results, MPI_Comm comm) {

  int rank, npes;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  int numPts = xPosArr.size();

  PetscInt* lx;
  PetscInt* ly;
  PetscInt* lz;
  PetscInt npx, npy, npz;
  PetscInt Ne;

  DAGetOwnershipRanges(dar1, &lx, &ly, &lz);
  DAGetInfo(dar1, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL, &npx, &npy, &npz,	
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  //Number of nodes is 1 more than the number of elements
  Ne--; 

  double h = 1.0/static_cast<double>(Ne);

  double* scanLx = new double[npx];
  double* scanLy = new double[npy];
  double* scanLz = new double[npz];

  scanLx[0] = 0;
  scanLy[0] = 0;
  scanLz[0] = 0;
  for(int i = 1; i < npx; i++) {
    scanLx[i] = scanLx[i-1] + (static_cast<double>(lx[i - 1])*h);
  }
  for(int i = 1; i < npy; i++) {
    scanLy[i] = scanLy[i-1] + (static_cast<double>(ly[i - 1])*h);
  }
  for(int i = 1; i < npz; i++) {
    scanLz[i] = scanLz[i-1] + (static_cast<double>(lz[i - 1])*h);
  }

  int* part = new int[numPts];
  int* sendCnts = new int[npes];
  int* recvCnts = new int[npes];
  int* sendOffsets = new int[npes];
  int* recvOffsets = new int[npes];

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
    recvCnts[i] = 0;
  }

  for(int i = 0; i < numPts; i++) {
    if( (xPosArr[i] > 0.0) && (yPosArr[i] > 0.0) && (zPosArr[i] > 0.0) &&
        (xPosArr[i] < 1.0) && (yPosArr[i] < 1.0) && (zPosArr[i] < 1.0) ) {
      unsigned int xRes, yRes, zRes;
      seq::maxLowerBound<double>(scanLx, xPosArr[i], xRes, 0, 0);
      seq::maxLowerBound<double>(scanLy, yPosArr[i], yRes, 0, 0);
      seq::maxLowerBound<double>(scanLz, zPosArr[i], zRes, 0, 0);
      part[i] = (((zRes*npy) + yRes)*npx) + xRes;
    } else {
      part[i] = rank;
    }
    sendCnts[part[i]] += 3;
  }

  par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, commAll);

  int totalRecv = 0;
  for(int i = 0; i < npes; i++) {
    totalRecv += recvCnts[i];
  }

  sendOffsets[0] = 0;
  recvOffsets[0] = 0;
  for(int i = 1; i < npes; i++) {
    sendOffsets[i] = sendOffsets[i - 1] + sendCnts[i - 1];
    recvOffsets[i] = recvOffsets[i - 1] + recvCnts[i - 1];
  }

  int* tmpSendCnts = new int[npes];  
  for(int i = 0; i < npes; i++) {
    tmpSendCnts[i] = 0;
  }

  int totalSend = (3*numPts);
  int* commMap = new int[numPts];  
  std::vector<double> xyzSendVals(totalSend);

  for(int i = 0; i < numPts; i++) {
    int idx = sendOffsets[part[i]] + tmpSendCnts[part[i]];
    tmpSendCnts[part[i]] += 3;
    xyzSendVals[idx] = xPosArr[i];
    xyzSendVals[idx + 1] = yPosArr[i];
    xyzSendVals[idx + 2] = zPosArr[i];
    commMap[i] = idx/3;
  }

  std::vector<double> xyzRecvVals(totalRecv);

  double* sendPtr = NULL;
  double* recvPtr = NULL;
  if(totalSend) {
    sendPtr = (&(*(xyzSendVals.begin())));
  }
  if(totalRecv) {
    recvPtr = (&(*(xyzRecvVals.begin()))); 
  }

  par::Mpi_Alltoallv_sparse<double>(sendPtr, sendCnts, sendOffsets,
      recvPtr, recvCnts, recvOffsets, commAll);

  PetscInt xs, ys, zs;
  DAGetCorners(dar1, &xs, &ys, &zs, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  double xOff = static_cast<double>(xs)*h;
  double yOff = static_cast<double>(ys)*h;
  double zOff = static_cast<double>(zs)*h;

  PetscScalar*** fArr;
  PetscScalar**** gArr;
  PetscScalar**** hArr;

  DAVecGetArray(dar1, fnLoc, &fArr);
  DAVecGetArrayDOF(dar2, gradLoc, &gArr);
  DAVecGetArrayDOF(dar3, hessLoc, &hArr);

  int numRecvPts = totalRecv/3;
  double* fVals = new double[totalRecv];
  for(int i = 0; i < numRecvPts; i++) {
    if( (xyzRecvVals[3*i] > 0) && (xyzRecvVals[(3*i) + 1] > 0) && (xyzRecvVals[(3*i) + 2] > 0) && 
        (xyzRecvVals[3*i] < 1) && (xyzRecvVals[(3*i) + 1] < 1) && (xyzRecvVals[(3*i) + 2] < 1) ) {
      double gradFnVals[3];
      evalQuinticGradFn(fArr, gArr, hArr, xOff, yOff, zOff,
          h, xyzRecvVals[3*i], xyzRecvVals[(3*i) + 1], xyzRecvVals[(3*i) + 2], gradFnVals);
      for(int j = 0; j < 3; j++) {
        fVals[(3*i) + j] = gradFnVals[j];
      }
    } else {
      for(int j = 0; j < 3; j++) {
        fVals[(3*i) + j] = 0;
      }
    }
  }

  DAVecRestoreArray(dar1, fnLoc, &fArr);
  DAVecRestoreArrayDOF(dar2, gradLoc, &gArr);
  DAVecRestoreArrayDOF(dar3, hessLoc, &hArr);

  double* recvFvals = new double[totalSend];

  par::Mpi_Alltoallv_sparse<double>(fVals, recvCnts, recvOffsets,
      recvFvals, sendCnts, sendOffsets, commAll);

  results.resize(totalSend);

  for(int i = 0; i < numPts; i++) {
    results[3*i] = recvFvals[3*commMap[i]];
    results[(3*i) + 1] = recvFvals[(3*commMap[i]) + 1];
    results[(3*i) + 2] = recvFvals[(3*commMap[i]) + 2];
  }

  delete [] recvFvals;
  delete [] fVals;
  delete [] commMap;
  delete [] tmpSendCnts;
  delete [] sendCnts;
  delete [] recvCnts;
  delete [] sendOffsets;
  delete [] recvOffsets;
  delete [] part;
  delete [] scanLx;
  delete [] scanLy;
  delete [] scanLz;  

}

void evalQuinticFnAtAllPts(std::vector<double>& xPosArr, std::vector<double>& yPosArr,
    std::vector<double>& zPosArr, DA dar1, DA dar2, DA dar3,
    Vec fnLoc, Vec gradLoc, Vec hessLoc, std::vector<double>& results, MPI_Comm comm) {

  int rank, npes;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  int numPts = xPosArr.size();

  PetscInt* lx;
  PetscInt* ly;
  PetscInt* lz;
  PetscInt npx, npy, npz;
  PetscInt Ne;

  DAGetOwnershipRanges(dar1, &lx, &ly, &lz);
  DAGetInfo(dar1, PETSC_NULL, &Ne, PETSC_NULL, PETSC_NULL, &npx, &npy, &npz,	
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  //Number of nodes is 1 more than the number of elements
  Ne--; 

  double h = 1.0/static_cast<double>(Ne);

  double* scanLx = new double[npx];
  double* scanLy = new double[npy];
  double* scanLz = new double[npz];

  scanLx[0] = 0;
  scanLy[0] = 0;
  scanLz[0] = 0;
  for(int i = 1; i < npx; i++) {
    scanLx[i] = scanLx[i-1] + (static_cast<double>(lx[i - 1])*h);
  }
  for(int i = 1; i < npy; i++) {
    scanLy[i] = scanLy[i-1] + (static_cast<double>(ly[i - 1])*h);
  }
  for(int i = 1; i < npz; i++) {
    scanLz[i] = scanLz[i-1] + (static_cast<double>(lz[i - 1])*h);
  }

  int* part = new int[numPts];
  int* sendCnts = new int[npes];
  int* recvCnts = new int[npes];
  int* sendOffsets = new int[npes];
  int* recvOffsets = new int[npes];

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
    recvCnts[i] = 0;
  }

  for(int i = 0; i < numPts; i++) {
    if( (xPosArr[i] > 0.0) && (yPosArr[i] > 0.0) && (zPosArr[i] > 0.0) &&
        (xPosArr[i] < 1.0) && (yPosArr[i] < 1.0) && (zPosArr[i] < 1.0) ) {
      unsigned int xRes, yRes, zRes;
      seq::maxLowerBound<double>(scanLx, xPosArr[i], xRes, 0, 0);
      seq::maxLowerBound<double>(scanLy, yPosArr[i], yRes, 0, 0);
      seq::maxLowerBound<double>(scanLz, zPosArr[i], zRes, 0, 0);
      part[i] = (((zRes*npy) + yRes)*npx) + xRes;
    } else {
      part[i] = rank;
    }
    sendCnts[part[i]] += 3;
  }

  par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, commAll);

  int totalRecv = 0;
  for(int i = 0; i < npes; i++) {
    totalRecv += recvCnts[i];
  }

  sendOffsets[0] = 0;
  recvOffsets[0] = 0;
  for(int i = 1; i < npes; i++) {
    sendOffsets[i] = sendOffsets[i - 1] + sendCnts[i - 1];
    recvOffsets[i] = recvOffsets[i - 1] + recvCnts[i - 1];
  }

  int* tmpSendCnts = new int[npes];  
  for(int i = 0; i < npes; i++) {
    tmpSendCnts[i] = 0;
  }

  int totalSend = (3*numPts);
  int* commMap = new int[numPts];  
  std::vector<double> xyzSendVals(totalSend);

  for(int i = 0; i < numPts; i++) {
    int idx = sendOffsets[part[i]] + tmpSendCnts[part[i]];
    tmpSendCnts[part[i]] += 3;
    xyzSendVals[idx] = xPosArr[i];
    xyzSendVals[idx + 1] = yPosArr[i];
    xyzSendVals[idx + 2] = zPosArr[i];
    commMap[i] = idx/3;
  }

  std::vector<double> xyzRecvVals(totalRecv);

  double* sendPtr = NULL;
  double* recvPtr = NULL;
  if(totalSend) {
    sendPtr = (&(*(xyzSendVals.begin())));
  }
  if(totalRecv) {
    recvPtr = (&(*(xyzRecvVals.begin()))); 
  }

  par::Mpi_Alltoallv_sparse<double>(sendPtr, sendCnts, sendOffsets,
      recvPtr, recvCnts, recvOffsets, commAll);

  PetscInt xs, ys, zs;
  DAGetCorners(dar1, &xs, &ys, &zs, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  double xOff = static_cast<double>(xs)*h;
  double yOff = static_cast<double>(ys)*h;
  double zOff = static_cast<double>(zs)*h;

  PetscScalar*** fArr;
  PetscScalar**** gArr;
  PetscScalar**** hArr;

  DAVecGetArray(dar1, fnLoc, &fArr);
  DAVecGetArrayDOF(dar2, gradLoc, &gArr);
  DAVecGetArrayDOF(dar3, hessLoc, &hArr);

  int numRecvPts = totalRecv/3;
  double* fVals = new double[numRecvPts];
  for(int i = 0; i < numRecvPts; i++) {
    if( (xyzRecvVals[3*i] > 0) && (xyzRecvVals[(3*i) + 1] > 0) && (xyzRecvVals[(3*i) + 2] > 0) && 
        (xyzRecvVals[3*i] < 1) && (xyzRecvVals[(3*i) + 1] < 1) && (xyzRecvVals[(3*i) + 2] < 1) ) {
      fVals[i] = evalQuinticFn(fArr, gArr, hArr, xOff, yOff, zOff,
          h, xyzRecvVals[3*i], xyzRecvVals[(3*i) + 1], xyzRecvVals[(3*i) + 2]);
    } else {
      fVals[i] = 0;
    }
  }

  DAVecRestoreArray(dar1, fnLoc, &fArr);
  DAVecRestoreArrayDOF(dar2, gradLoc, &gArr);
  DAVecRestoreArrayDOF(dar3, hessLoc, &hArr);

  double* recvFvals = new double[numPts];
  for(int i = 0; i < npes; i++) {
    sendCnts[i] = sendCnts[i]/3;
    recvCnts[i] = recvCnts[i]/3;
  }

  sendOffsets[0] = 0;
  recvOffsets[0] = 0;
  for(int i = 1; i < npes; i++) {
    sendOffsets[i] = sendOffsets[i - 1] + sendCnts[i - 1];
    recvOffsets[i] = recvOffsets[i - 1] + recvCnts[i - 1];
  }

  par::Mpi_Alltoallv_sparse<double>(fVals, recvCnts, recvOffsets,
      recvFvals, sendCnts, sendOffsets, commAll);

  results.resize(numPts);

  for(int i = 0; i < numPts; i++) {
    results[i] = recvFvals[commMap[i]];
  }

  delete [] recvFvals;
  delete [] fVals;
  delete [] commMap;
  delete [] tmpSendCnts;
  delete [] sendCnts;
  delete [] recvCnts;
  delete [] sendOffsets;
  delete [] recvOffsets;
  delete [] part;
  delete [] scanLx;
  delete [] scanLy;
  delete [] scanLz;  

}

//Each processor owns a cuboidal portion of the data.
//xPos, yPos, zPos are the global coordinates
//There is no communication here, because we assume
//the shared node values for f, gf1,... are duplicated.
double evalQuinticFn(const PetscScalar*** fArr, const PetscScalar**** gArr, 
    const PetscScalar**** hArr, double xOff, double yOff, double zOff,
    double h, double xPos, double yPos, double zPos) {

  double res;

  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    res = 0.0;
  } else {   
    int ei = static_cast<int>((xPos - xOff)/h);
    int ej = static_cast<int>((yPos - yOff)/h);
    int ek = static_cast<int>((zPos - zOff)/h);
    double x0 = static_cast<double>(ei)*h;
    double y0 = static_cast<double>(ej)*h;
    double z0 = static_cast<double>(ek)*h;
    double psi = ((xPos - xOff - x0)*2.0/h) - 1.0;
    double eta = ((yPos - yOff - y0)*2.0/h) - 1.0;
    double gamma = ((zPos - zOff - z0)*2.0/h) - 1.0;
    double phiVals[8][10];

    evalAll3Dquintic(psi, eta, gamma, phiVals);

    res = 0.0;
    for(int i = 0; i < 8; i++) {
      int xid = ei + (i%2);
      int yid = ej + ((i/2)%2);
      int zid = ek + (i/4);
      res += (fArr[zid][yid][xid]*phiVals[i][0]);
      for(int j = 0; j < 3; j++) {
        res += (0.5*h*gArr[zid][yid][xid][j]*phiVals[i][1 + j]);
      }
      for(int j = 0; j < 6; j++) {
        res += (0.25*h*h*hArr[zid][yid][xid][j]*phiVals[i][4 + j]);
      }
    }
  }

  return res;
}

void evalQuinticGradFn(const PetscScalar*** fArr, const PetscScalar**** gArr, 
    const PetscScalar**** hArr, double xOff, double yOff, double zOff,
    double h, double xPos, double yPos, double zPos, double* res) {

  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    res[0] = 0.0;
    res[1] = 0.0;
    res[2] = 0.0;
  } else {   
    int ei = static_cast<int>((xPos - xOff)/h);
    int ej = static_cast<int>((yPos - yOff)/h);
    int ek = static_cast<int>((zPos - zOff)/h);
    double x0 = static_cast<double>(ei)*h;
    double y0 = static_cast<double>(ej)*h;
    double z0 = static_cast<double>(ek)*h;
    double psi = ((xPos - xOff - x0)*2.0/h) - 1.0;
    double eta = ((yPos - yOff - y0)*2.0/h) - 1.0;
    double gamma = ((zPos - zOff - z0)*2.0/h) - 1.0;
    double gradPhiVals[8][10][3];

    evalAll3DquinticGrad(psi, eta, gamma, gradPhiVals);

    res[0] = 0.0;
    res[1] = 0.0;
    res[2] = 0.0;
    for(int i = 0; i < 8; i++) {
      int xid = ei + (i%2);
      int yid = ej + ((i/2)%2);
      int zid = ek + (i/4);
      for(int j = 0; j < 3; j++) {
        res[j] += (fArr[zid][yid][xid]*gradPhiVals[i][0][j]);
        for(int k = 0; k < 3; k++) {
          res[j] += (0.5*h*gArr[zid][yid][xid][k]*gradPhiVals[i][1 + k][j]);
        }
        for(int k = 0; k < 6; k++) {
          res[j] += (0.25*h*h*hArr[zid][yid][xid][k]*gradPhiVals[i][4 + k][j]);
        }
      }
    }
  }
}

void evalAll3Dquintic(double psi, double eta, double gamma, double** phiArr) {
  int psiMap = {0, 1, 0, 1, 0, 1, 0, 1};
  int etaMap = {0, 0, 1, 1, 0, 0, 1, 1};
  int gammaMap = {0, 0, 0, 0, 1, 1, 1, 1};
  for(int i = 0; i < 8; i++) {
    phiArr[i][0] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));

    phiArr[i][1] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));

    phiArr[i][2] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));

    phiArr[i][3] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));

    phiArr[i][4] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));

    phiArr[i][5] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));

    phiArr[i][6] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));

    phiArr[i][7] = (eval1Dquintic(psiMap[i], 2, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));

    phiArr[i][8] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 2, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));

    phiArr[i][9] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 2, gamma));
  }
}

void evalAll3DquinticGrad(double psi, double eta, double gamma, double*** gradPhiArr) {
  int psiMap = {0, 1, 0, 1, 0, 1, 0, 1};
  int etaMap = {0, 0, 1, 1, 0, 0, 1, 1};
  int gammaMap = {0, 0, 0, 0, 1, 1, 1, 1};
  for(int i = 0; i < 8; i++) {
    gradPhiArr[i][0][0] = (eval1DquinticGrad(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][0][1] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1DquinticGrad(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][0][2] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1DquinticGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][1][0] = (eval1DquinticGrad(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][1][1] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1DquinticGrad(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][1][2] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1DquinticGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][2][0] = (eval1DquinticGrad(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][2][1] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1DquinticGrad(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][2][2] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1DquinticGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][3][0] = (eval1DquinticGrad(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));
    gradPhiArr[i][3][1] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1DquinticGrad(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));
    gradPhiArr[i][3][2] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1DquinticGrad(gammaMap[i], 1, gamma));

    gradPhiArr[i][4][0] = (eval1DquinticGrad(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][4][1] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1DquinticGrad(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][4][2] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1DquinticGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][5][0] = (eval1DquinticGrad(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));
    gradPhiArr[i][5][1] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1DquinticGrad(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));
    gradPhiArr[i][5][2] = (eval1Dquintic(psiMap[i], 1, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1DquinticGrad(gammaMap[i], 1, gamma));

    gradPhiArr[i][6][0] = (eval1DquinticGrad(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));
    gradPhiArr[i][6][1] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1DquinticGrad(etaMap[i], 1, eta)*
        eval1Dquintic(gammaMap[i], 1, gamma));
    gradPhiArr[i][6][2] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 1, eta)*
        eval1DquinticGrad(gammaMap[i], 1, gamma));

    gradPhiArr[i][7][0] = (eval1DquinticGrad(psiMap[i], 2, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][7][1] = (eval1Dquintic(psiMap[i], 2, psi)*
        eval1DquinticGrad(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][7][2] = (eval1Dquintic(psiMap[i], 2, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1DquinticGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][8][0] = (eval1DquinticGrad(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 2, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][8][1] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1DquinticGrad(etaMap[i], 2, eta)*
        eval1Dquintic(gammaMap[i], 0, gamma));
    gradPhiArr[i][8][2] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 2, eta)*
        eval1DquinticGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][9][0] = (eval1DquinticGrad(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 2, gamma));
    gradPhiArr[i][9][1] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1DquinticGrad(etaMap[i], 0, eta)*
        eval1Dquintic(gammaMap[i], 2, gamma));
    gradPhiArr[i][9][2] = (eval1Dquintic(psiMap[i], 0, psi)*
        eval1Dquintic(etaMap[i], 0, eta)*
        eval1DquinticGrad(gammaMap[i], 2, gamma));

  }
}

double eval1Dquintic(int nodeNum, int compNum, double psi)
{
  double phi;

  switch(nodeNum) {
    case 0: {
              switch(compNum) {
                case 0: {
                          phi = -(1.0/16.0)*(CUBE(psi - 1.0))*((3.0*(SQR(psi))) + (9.0*psi) + 8.0);
                          break;
                        }
                case 1: {
                          phi = -(1.0/16.0)*(CUBE(psi - 1.0))*((3.0*psi) + 5.0)*(psi + 1.0);
                          break;
                        }
                case 2: {
                          phi = -(1.0/16.0)*(CUBE(psi - 1.0))*(SQR(psi + 1.0));
                          break;
                        }
                default: {
                           std::cerr<< "nodeNum: wrong argument"<<std::endl;
                           exit(1);
                         }
              }
              break;
            }
    case 1: {
              switch(compNum) {
                case 0: {
                          phi = (1.0/16.0)*(CUBE(psi + 1.0))*((3.0*(SQR(psi))) - (9.0*psi) + 8.0);
                          break;
                        }
                case 1: {
                          phi = -(1.0/16.0)*(CUBE(psi + 1.0))*((3.0*psi) - 5.0)*(psi - 1.0);
                          break;
                        }
                case 2: {
                          phi = (1.0/16.0)*(CUBE(psi + 1.0))*(SQR(psi - 1.0));
                          break;
                        }
                default: {
                           std::cerr<< "nodeNum: wrong argument"<<std::endl;
                           exit(1);
                         }
              }
              break;
            }
    default: {
               std::cerr<< "nodeNum: wrong argument"<<std::endl;
               exit(1);
             }
  }

  return phi;
}

double eval1DquinticGrad(int nodeNum, int compNum, double psi)
{
  double phi;

  switch(nodeNum) {
    case 0: {
              switch(compNum) {
                case 0: {
                          phi =  -(3.0/16.0)*(SQR(psi - 1.0))*((3.0*(SQR(psi))) + (9.0*psi) + 8.0)
                            - (1.0/16.0)*(CUBE(psi - 1.0))*((6.0*psi) + 9.0);
                          break;
                        }
                case 1: {
                          phi = -(3.0/16.0)*(SQR(psi - 1.0))*((3.0*psi) + 5.0)*(psi + 1.0) -
                            (3.0/16.0)*(CUBE(psi - 1.0))*(psi + 1.0) -
                            (1.0/16.0)*(CUBE(psi - 1.0))*((3.0*psi) + 5.0);
                          break;
                        }
                case 2: {
                          phi = -(3.0/16.0)*(SQR(psi - 1.0))*(SQR(psi + 1.0)) -
                            (1.0/8.0)*(CUBE(psi - 1.0))*(psi + 1.0);
                          break;
                        }
                default: {
                           std::cerr<<"nodeNum: wrong argument"<<std::endl;
                           exit(1);
                         }
              }
              break;
            }
    case 1: {
              switch(compNum) {
                case 0: {
                          phi = (3.0/16.0)*(SQR(psi + 1.0))*((3.0*(SQR(psi))) -
                              (9.0*psi) + 8.0) + (1.0/16.0)*(CUBE(psi + 1.0))*((6.0*psi) - 9.0);
                          break;
                        }
                case 1: {
                          phi =  -(3.0/16.0)*(SQR(psi + 1.0))*((3.0*psi) - 5.0)*(psi - 1.0) -
                            (3.0/16.0)*(CUBE(psi + 1.0))*(psi - 1.0) - 
                            (1.0/16.0)*(CUBE(psi + 1.0))*((3.0*psi) - 5.0);
                          break;
                        }
                case 2: {
                          phi =  (3.0/16.0)*(SQR(psi - 1.0))*(SQR(psi + 1.0)) +
                            (1.0/8.0)*(CUBE(psi + 1.0))*(psi - 1.0);
                          break;
                        }
                default: {
                           std::cerr<<"nodeNum: wrong argument"<<std::endl;
                           exit(1);
                         }
              }
              break;
            }
    default: {
               std::cerr<<"nodeNum: wrong argument"<<std::endl;
               exit(1);
             }
  }

  return phi;
}







