
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
#include "regInterpCubic.h"

extern int computeSigEvent;
extern int computeTauEvent;
extern int computeGradTauEvent;
extern int computeNodalTauEvent;
extern int computeNodalGradTauEvent;

#ifndef CUBE
#define CUBE(a) ((a)*(a)*(a))
#endif

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

void computeNewSigVals(ot::DA* dao, double patchWidth,
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& sigLocal,
    const std::vector<std::vector<double> >& gradSigLocal, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals) {

  PetscLogEventBegin(computeSigEvent, 0, 0, 0, 0);

  assert(dao != NULL);

  unsigned int elemSize = dao->getElementSize();

  if(!(dao->iAmActive())) {
    assert(elemSize == 0);
    assert(imgPatches.empty());
  }

  if(imgPatches.empty()) {
    assert(elemSize == 0);
  }

  int numImages = sigLocal.size();
  assert(numImages);
  assert(gradSigLocal.size() == numImages);

  sigVals.resize(numImages);
  for(int i = 0; i < numImages; i++) {    
    sigVals[i].resize(numGpts*numGpts*numGpts*elemSize);
  }

  if(dao->iAmActive()) {
    unsigned int maxDepth = dao->getMaxDepth();
    double hFac = 1.0/static_cast<double>(1u << (maxDepth - 1));
    unsigned int elemCtr = 0;
    for(dao->init<ot::DA_FLAGS::WRITABLE>();
        dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
        dao->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = dao->getCurrentOffset();
      unsigned int idx = dao->curr();
      unsigned int lev = dao->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      double hOct = hFac*(static_cast<double>(1u << (maxDepth - lev)));
      double x0 = hFac*static_cast<double>(xint);
      double y0 = hFac*static_cast<double>(yint);
      double z0 = hFac*static_cast<double>(zint);
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            double xPos = (x0 + (0.5*hOct*(1.0 + gPts[m])));
            double yPos = (y0 + (0.5*hOct*(1.0 + gPts[n])));
            double zPos = (z0 + (0.5*hOct*(1.0 + gPts[p])));
            for(int i = 0; i < numImages; i++) {
              assert( ((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p) < sigVals[i].size() );
              sigVals[i][(((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p] =
                newEvalCubicFn(sigLocal[i], gradSigLocal[i],
                    imgPatches, mesh, xPos, yPos, zPos);
            }//end for i
          }//end for p
        }//end for n
      }//end for m
      elemCtr++;
    }//end WRITABLE
    assert(elemCtr == elemSize);
  }//end if active

  PetscLogEventEnd(computeSigEvent, 0, 0, 0, 0);
}


void computeNewNodalGradTauAtU(ot::DA* dao, double patchWidth,
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal, Vec U,
    std::vector<std::vector<double> >& nodalGradTauAtU) {

  PetscLogEventBegin(computeNodalGradTauEvent, 0, 0, 0, 0);

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  assert(dao != NULL);

  int rank = dao->getRankAll();

  std::vector<unsigned int> outOfBndsList;
  newComposeXposAtUnodal(dao, patchWidth, U, xPosArr, yPosArr, zPosArr, outOfBndsList);

  //evaluate at all points
  newEvalGradFnAtAllPts(dao, xPosArr, yPosArr, zPosArr, outOfBndsList, imgPatches,
      mesh, tauLocal, gradTauLocal, nodalGradTauAtU);

  PetscLogEventEnd(computeNodalGradTauEvent, 0, 0, 0, 0);
}


void computeNewNodalTauAtU(ot::DA* dao, double patchWidth, 
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal,
    Vec U, std::vector<std::vector<double> >& nodalTauAtU) {

  PetscLogEventBegin(computeNodalTauEvent, 0, 0, 0, 0);

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  std::vector<unsigned int> outOfBndsList;
  newComposeXposAtUnodal(dao, patchWidth, U, xPosArr, yPosArr, zPosArr, outOfBndsList);

  //evaluate at all points
  newEvalFnAtAllPts(dao, xPosArr, yPosArr, zPosArr, outOfBndsList, imgPatches,
      mesh, tauLocal, gradTauLocal, nodalTauAtU);

  PetscLogEventEnd(computeNodalTauEvent, 0, 0, 0, 0);
}

void computeNewTauAtU(ot::DA* dao, double patchWidth,
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> > & tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU) {

  PetscLogEventBegin(computeTauEvent, 0, 0, 0, 0);

  int numImages = tauLocal.size();

  assert(numImages);
  assert(dao != NULL);

  unsigned int elemSize = dao->getElementSize();

  if(!(dao->iAmActive())) {
    assert(elemSize == 0);
    assert(imgPatches.empty());
  }

  if(imgPatches.empty()) {
    assert(elemSize == 0);
  }

  tauAtU.resize(numImages);
  for(int i = 0; i < numImages; i++) {
    tauAtU[i].resize(numGpts*numGpts*numGpts*elemSize);
  }

  typedef double* doublePtr;
  typedef double** double2Ptr;

  if(dao->iAmActive()) {
    std::vector<unsigned int> outOfBndsList;
    std::vector<double> xPosArr;
    std::vector<double> yPosArr;
    std::vector<double> zPosArr;

    unsigned int maxDepth = dao->getMaxDepth();
    double hFac = 1.0/static_cast<double>(1u << (maxDepth - 1));

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
      }//end n
    }//end m

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

    unsigned int elemCtr = 0;
    for(dao->init<ot::DA_FLAGS::WRITABLE>();
        dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
        dao->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = dao->getCurrentOffset();
      unsigned int idx = dao->curr();
      unsigned int lev = dao->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      double hOct = hFac*(static_cast<double>(1u << (maxDepth - lev)));
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
            }//end k
          }//end p
        }//end n
      }//end m
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            double xPos = (x0 + (0.5*hOct*(1.0 + gPts[m])) + u1loc[m][n][p]);
            double yPos = (y0 + (0.5*hOct*(1.0 + gPts[n])) + u2loc[m][n][p]);
            double zPos = (z0 + (0.5*hOct*(1.0 + gPts[p])) + u3loc[m][n][p]);
            if( (fabs(u1loc[m][n][p]) >= patchWidth) ||
                (fabs(u2loc[m][n][p]) >= patchWidth) ||
                (fabs(u3loc[m][n][p]) >= patchWidth) ) {
              outOfBndsList.push_back((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p);
              xPosArr.push_back(xPos);
              yPosArr.push_back(yPos);
              zPosArr.push_back(zPos);
            } else {
              for(int i = 0; i < numImages; i++) {
                assert( ((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p) < tauAtU[i].size() );
                tauAtU[i][(((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p] = 
                  newEvalCubicFn(tauLocal[i], gradTauLocal[i],
                      imgPatches, mesh, xPos, yPos, zPos);
              }//end i
            }//end if large disp
          }//end p
        }//end n
      }//end m
      elemCtr++;
    }//end WRITABLE

    assert(elemCtr == elemSize);

    dao->vecRestoreBuffer(U, uArr, false, false, true, 3);

    for(int m = 0; m < numGpts; m++){
      for(int n = 0; n < numGpts; n++){
        delete [] u1loc[m][n];
        u1loc[m][n] = NULL;

        delete [] u2loc[m][n];
        u2loc[m][n] = NULL;

        delete [] u3loc[m][n];
        u3loc[m][n] = NULL;
      }//end n
      delete [] u1loc[m];
      u1loc[m] = NULL;

      delete [] u2loc[m];
      u2loc[m] = NULL;

      delete [] u3loc[m];
      u3loc[m] = NULL;
    }//end m
    delete [] u1loc;
    u1loc = NULL;

    delete [] u2loc;
    u2loc = NULL;

    delete [] u3loc;
    u3loc = NULL;

    //Remote Portion
    std::vector<ot::TreeNode> minBlocks = dao->getMinAllBlocks();

    int rankActive = dao->getRankActive();
    int npesActive = dao->getNpesActive();
    MPI_Comm commActive = dao->getCommActive();

    double fac = static_cast<double>((1u << (maxDepth - 1)));

    unsigned int* part = NULL;
    if(!(outOfBndsList.empty())) {
      part = new unsigned int[outOfBndsList.size()]; 
      assert(part);
    }
    int* sendSizes = new int[npesActive];
    assert(sendSizes);
    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = 0;
    }//end for i

    assert(xPosArr.size() == outOfBndsList.size());
    assert(yPosArr.size() == outOfBndsList.size());
    assert(zPosArr.size() == outOfBndsList.size());

    for(int l = 0; l < outOfBndsList.size(); l++) {
      double xPos = xPosArr[l];
      double yPos = yPosArr[l];
      double zPos = zPosArr[l];
      if( (xPos < 0.0) || (xPos >= 1.0) || 
          (yPos < 0.0) || (yPos >= 1.0) || 
          (zPos < 0.0) || (zPos >= 1.0) ) {
        part[l] = rankActive;
      } else {
        unsigned int xint = static_cast<unsigned int>(fac*xPos);
        unsigned int yint = static_cast<unsigned int>(fac*yPos);
        unsigned int zint = static_cast<unsigned int>(fac*zPos);

        ot::TreeNode tmpNode(xint, yint, zint, maxDepth, 3, maxDepth);
        bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks, tmpNode, part[l], NULL, NULL);
        assert(found);
      }
      assert(part[l] < npesActive);
      sendSizes[part[l]] += 3;
    }//end for l

    int* recvSizes = new int[npesActive];
    assert(recvSizes);

    par::Mpi_Alltoall<int>(sendSizes, recvSizes, 1, commActive);

    int* sendOff = new int[npesActive];
    assert(sendOff);

    int* recvOff = new int[npesActive];
    assert(recvOff);

    sendOff[0] = 0;
    recvOff[0] = 0;
    for(int i = 1; i < npesActive; i++) {
      sendOff[i] = sendOff[i - 1] + sendSizes[i - 1];
      recvOff[i] = recvOff[i - 1] + recvSizes[i - 1];
    }//end for i

    assert( (3*(outOfBndsList.size())) == (sendOff[npesActive - 1] + sendSizes[npesActive - 1]) );

    double* sendPts = NULL;
    if(!(outOfBndsList.empty())) {
      sendPts = new double[3*(outOfBndsList.size())];
      assert(sendPts);
    }

    double* recvPts = NULL;
    if( recvOff[npesActive - 1] + recvSizes[npesActive - 1] ) {
      recvPts = new double[recvOff[npesActive - 1] + recvSizes[npesActive - 1]];
      assert(recvPts);
    }

    int* tmpSendSizes = new int[npesActive];
    assert(tmpSendSizes);
    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      double xPos = xPosArr[l];
      double yPos = yPosArr[l];
      double zPos = zPosArr[l];
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      sendPts[ptIdx] = xPos;
      sendPts[ptIdx + 1] = yPos;
      sendPts[ptIdx + 2] = zPos;
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += 3;
    }//end for l

    par::Mpi_Alltoallv_sparse<double>(sendPts, sendSizes, sendOff,
        recvPts, recvSizes, recvOff, commActive);

    unsigned int totalRecvPts = (recvOff[npesActive - 1] + recvSizes[npesActive-1])/3;
    double* tmpSendResults = NULL;
    if(totalRecvPts) {
      tmpSendResults = new double[numImages*totalRecvPts];
      assert(tmpSendResults);
    }

    if(totalRecvPts) {
      assert(!(imgPatches.empty()));
      assert(!(mesh.empty()));
      for(int i = 0; i < numImages; i++) {
        assert(!(tauLocal[i].empty()));
        assert(!(gradTauLocal[i].empty()));
      }//end for i
    }

    for(int j = 0; j < totalRecvPts; j++) {
      double xPos = recvPts[(3*j)];
      double yPos = recvPts[(3*j) + 1];
      double zPos = recvPts[(3*j) + 2];
      for(int i = 0; i < numImages; i++) {
        tmpSendResults[(numImages*j) + i] = newEvalCubicFn(tauLocal[i], gradTauLocal[i],
            imgPatches, mesh, xPos, yPos, zPos);
      }//end for i
    }//end for j

    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = (numImages*(sendSizes[i]/3));
      recvSizes[i] = (numImages*(recvSizes[i]/3));
      sendOff[i] = (numImages*(sendOff[i]/3));
      recvOff[i] = (numImages*(recvOff[i]/3));
    }//end for i

    double* tmpRecvResults = NULL;
    if(!(outOfBndsList.empty())) {
      tmpRecvResults = new double[numImages*(outOfBndsList.size())];
      assert(tmpRecvResults);
    }

    par::Mpi_Alltoallv_sparse<double>(tmpSendResults, recvSizes, recvOff,
        tmpRecvResults, sendSizes, sendOff, commActive);

    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      for(int i = 0; i < numImages; i++) {
        assert( outOfBndsList[l] < tauAtU[i].size() );
        assert( (ptIdx + i) < (numImages*(outOfBndsList.size())) );
        tauAtU[i][outOfBndsList[l]] = tmpRecvResults[ptIdx + i];
      }//end for i
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += numImages;
    }//end for l

    if(tmpSendResults) {
      delete [] tmpSendResults;
      tmpSendResults = NULL;
    }

    if(tmpRecvResults) {
      delete [] tmpRecvResults;
      tmpRecvResults = NULL;
    }

    if(part) {
      delete [] part;
      part = NULL;
    }

    if(sendPts) {
      delete [] sendPts;
      sendPts = NULL;
    }

    if(recvPts) {
      delete [] recvPts;
      recvPts = NULL;
    }

    assert(tmpSendSizes != NULL);
    delete [] tmpSendSizes;
    tmpSendSizes = NULL;

    assert(sendOff != NULL);
    delete [] sendOff;
    sendOff = NULL;

    assert(recvOff != NULL);
    delete [] recvOff;
    recvOff = NULL;

    assert(sendSizes != NULL);
    delete [] sendSizes;
    sendSizes = NULL;

    assert(recvSizes != NULL);
    delete [] recvSizes;
    recvSizes = NULL;

  }//end if active

  PetscLogEventEnd(computeTauEvent, 0, 0, 0, 0);
}


void computeNewGradTauAtU(ot::DA* dao, double patchWidth,
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> > & tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU) { 

  PetscLogEventBegin(computeGradTauEvent, 0, 0, 0, 0);

  assert(dao != NULL);

  unsigned int elemSize = dao->getElementSize();

  if(!(dao->iAmActive())) {
    assert(elemSize == 0);
    assert(imgPatches.empty());
  }

  int numImages = tauLocal.size();
  assert(numImages);

  gradTauAtU.resize(numImages);
  for(int i = 0; i < numImages; i++) {
    gradTauAtU[i].resize(3*numGpts*numGpts*numGpts*elemSize);
  }//end for i

  unsigned int Ne;
  if(!(imgPatches.empty())) {
    unsigned int regLev = imgPatches[0].getLevel();
    Ne = (1u << (regLev - 1));
  } else {
    assert(elemSize == 0);
  }

  typedef double* doublePtr;
  typedef double** double2Ptr;

  if(dao->iAmActive()) {
    std::vector<unsigned int> outOfBndsList;
    std::vector<double> xPosArr;
    std::vector<double> yPosArr;
    std::vector<double> zPosArr;

    unsigned int maxDepth = dao->getMaxDepth();
    double hFac = 1.0/static_cast<double>(1u << (maxDepth - 1));

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
      }//end n
    }//end m

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

    unsigned int elemCtr = 0;
    for(dao->init<ot::DA_FLAGS::WRITABLE>();
        dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
        dao->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = dao->getCurrentOffset();
      unsigned int idx = dao->curr();
      unsigned int lev = dao->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      double hOct = hFac*(static_cast<double>(1u << (maxDepth - lev)));
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
            }//end k
          }//end p
        }//end n
      }//end m
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            double xPos = (x0 + (0.5*hOct*(1.0 + gPts[m])) + u1loc[m][n][p]);
            double yPos = (y0 + (0.5*hOct*(1.0 + gPts[n])) + u2loc[m][n][p]);
            double zPos = (z0 + (0.5*hOct*(1.0 + gPts[p])) + u3loc[m][n][p]);
            if( (fabs(u1loc[m][n][p]) >= patchWidth) ||
                (fabs(u2loc[m][n][p]) >= patchWidth) ||
                (fabs(u3loc[m][n][p]) >= patchWidth) ) {
              outOfBndsList.push_back((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p);
              xPosArr.push_back(xPos);
              yPosArr.push_back(yPos);
              zPosArr.push_back(zPos);
            } else {
              for(int i = 0; i < numImages; i++) {
                double res[3];
                newEvalCubicGradFn(tauLocal[i], gradTauLocal[i], imgPatches, 
                    mesh, xPos, yPos, zPos, res);
                for(int k = 0; k < 3; k++) {
                  assert( ((((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p)*3) + k) < gradTauAtU[i].size() );
                  gradTauAtU[i][(((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p)*3) + k] 
                    = (res[k]*2.0*static_cast<double>(Ne));
                }//end k
              }//end i
            }//end if large disp
          }//end p
        }//end n
      }//end m
      elemCtr++;
    }//end WRITABLE

    assert(elemCtr == elemSize);

    dao->vecRestoreBuffer(U, uArr, false, false, true, 3);

    for(int m = 0; m < numGpts; m++){
      for(int n = 0; n < numGpts; n++){
        delete [] u1loc[m][n];
        u1loc[m][n] = NULL;

        delete [] u2loc[m][n];
        u2loc[m][n] = NULL;

        delete [] u3loc[m][n];
        u3loc[m][n] = NULL;
      }//end n
      delete [] u1loc[m];
      u1loc[m] = NULL;

      delete [] u2loc[m];
      u2loc[m] = NULL;

      delete [] u3loc[m];
      u3loc[m] = NULL;
    }//end m
    delete [] u1loc;
    u1loc = NULL;

    delete [] u2loc;
    u2loc = NULL;

    delete [] u3loc;
    u3loc = NULL;

    //Remote Portion
    std::vector<ot::TreeNode> minBlocks = dao->getMinAllBlocks();

    int rankActive = dao->getRankActive();
    int npesActive = dao->getNpesActive();
    MPI_Comm commActive = dao->getCommActive();

    double fac = static_cast<double>((1u << (maxDepth - 1)));

    unsigned int* part = NULL;
    if(!(outOfBndsList.empty())) {
      part = new unsigned int[outOfBndsList.size()]; 
      assert(part);
    }

    int* sendSizes = new int[npesActive];
    assert(sendSizes);
    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = 0;
    }//end for i

    assert(xPosArr.size() == outOfBndsList.size());
    assert(yPosArr.size() == outOfBndsList.size());
    assert(zPosArr.size() == outOfBndsList.size());

    for(int l = 0; l < outOfBndsList.size(); l++) {
      double xPos = xPosArr[l];
      double yPos = yPosArr[l];
      double zPos = zPosArr[l];
      if( (xPos < 0.0) || (xPos >= 1.0) || 
          (yPos < 0.0) || (yPos >= 1.0) || 
          (zPos < 0.0) || (zPos >= 1.0) ) {
        part[l] = rankActive;
      } else {
        unsigned int xint = static_cast<unsigned int>(fac*xPos);
        unsigned int yint = static_cast<unsigned int>(fac*yPos);
        unsigned int zint = static_cast<unsigned int>(fac*zPos);

        ot::TreeNode tmpNode(xint, yint, zint, maxDepth, 3, maxDepth);
        bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks, tmpNode, part[l], NULL, NULL);
        assert(found);
      }
      assert(part[l] < npesActive);
      sendSizes[part[l]] += 3;
    }//end for l

    int* recvSizes = new int[npesActive];
    assert(recvSizes);

    par::Mpi_Alltoall<int>(sendSizes, recvSizes, 1, commActive);

    int* sendOff = new int[npesActive];
    assert(sendOff);

    int* recvOff = new int[npesActive];
    assert(recvOff);

    sendOff[0] = 0;
    recvOff[0] = 0;
    for(int i = 1; i < npesActive; i++) {
      sendOff[i] = sendOff[i - 1] + sendSizes[i - 1];
      recvOff[i] = recvOff[i - 1] + recvSizes[i - 1];
    }//end for i

    assert( (3*(outOfBndsList.size())) == (sendOff[npesActive - 1] + sendSizes[npesActive - 1]) );

    double* sendPts = NULL;
    if(!(outOfBndsList.empty())) {
      sendPts = new double[3*(outOfBndsList.size())];
      assert(sendPts);
    }

    double* recvPts = NULL;
    if( recvOff[npesActive - 1] + recvSizes[npesActive - 1] ) {
      recvPts = new double[recvOff[npesActive - 1] + recvSizes[npesActive - 1]];
      assert(recvPts);
    }

    int* tmpSendSizes = new int[npesActive];
    assert(tmpSendSizes);
    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      double xPos = xPosArr[l];
      double yPos = yPosArr[l];
      double zPos = zPosArr[l];
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      sendPts[ptIdx] = xPos;
      sendPts[ptIdx + 1] = yPos;
      sendPts[ptIdx + 2] = zPos;
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += 3;
    }//end for l

    par::Mpi_Alltoallv_sparse<double>(sendPts, sendSizes, sendOff,
        recvPts, recvSizes, recvOff, commActive);

    unsigned int totalRecvPts = (recvOff[npesActive - 1] + recvSizes[npesActive-1])/3;
    double* tmpSendResults = NULL;

    if(totalRecvPts) {
      tmpSendResults = new double[3*numImages*totalRecvPts];
      assert(tmpSendResults);
    }

    if(totalRecvPts) {
      assert(!(imgPatches.empty()));
      assert(!(mesh.empty()));
    }

    for(int j = 0; j < totalRecvPts; j++) {
      double xPos = recvPts[(3*j)];
      double yPos = recvPts[(3*j) + 1];
      double zPos = recvPts[(3*j) + 2];
      for(int i = 0; i < numImages; i++) {
        double res[3];
        newEvalCubicGradFn(tauLocal[i], gradTauLocal[i], imgPatches, 
            mesh, xPos, yPos, zPos, res);
        for(int k = 0; k < 3; k++) {
          tmpSendResults[(((numImages*j) + i)*3) + k] = (res[k]*2.0*static_cast<double>(Ne));
        }//end k
      }//end i
    }//end for j

    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = (numImages*(sendSizes[i]));
      recvSizes[i] = (numImages*(recvSizes[i]));
      sendOff[i] = (numImages*(sendOff[i]));
      recvOff[i] = (numImages*(recvOff[i]));
    }//end for i

    double* tmpRecvResults = NULL;
    if(!(outOfBndsList.empty())) {
      tmpRecvResults = new double[3*numImages*(outOfBndsList.size())];
      assert(tmpRecvResults);
    }

    par::Mpi_Alltoallv_sparse<double>(tmpSendResults, recvSizes, recvOff,
        tmpRecvResults, sendSizes, sendOff, commActive);

    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      for(int i = 0; i < numImages; i++) {
        for(int k = 0; k < 3; k++) {
          assert( ((3*outOfBndsList[l]) + k) < gradTauAtU[i].size() );
          assert( (ptIdx + (3*i) + k) < (3*numImages*(outOfBndsList.size())) );
          gradTauAtU[i][(3*outOfBndsList[l]) + k] = tmpRecvResults[ptIdx + (3*i) + k];
        }//end k
      }//end for i
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += (3*numImages);
    }//end for l

    if(tmpSendResults) {
      delete [] tmpSendResults;
      tmpSendResults = NULL;
    }

    if(tmpRecvResults) {
      delete [] tmpRecvResults;
      tmpRecvResults = NULL;
    }

    if(part) {
      delete [] part;
      part = NULL;
    }

    if(sendPts) {
      delete [] sendPts;
      sendPts = NULL;
    }

    if(recvPts) {
      delete [] recvPts;
      recvPts = NULL;
    }

    assert(tmpSendSizes != NULL);
    delete [] tmpSendSizes;
    tmpSendSizes = NULL;

    assert(sendOff != NULL);
    delete [] sendOff;
    sendOff = NULL;

    assert(recvOff != NULL);
    delete [] recvOff;
    recvOff = NULL;

    assert(sendSizes != NULL);
    delete [] sendSizes;
    sendSizes = NULL;

    assert(recvSizes != NULL);
    delete [] recvSizes;
    recvSizes = NULL;

  }//end if active

  PetscLogEventEnd(computeGradTauEvent, 0, 0, 0, 0);
}

void newEvalFnAtAllPts(ot::DA* dao, const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    const std::vector<unsigned int>& outOfBndsList,
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results) {

  int numImages = F.size();
  int numPts = xPosArr.size();

  assert(numImages);

  results.resize(numImages);
  for(int i = 0; i < numImages; i++) {
    results[i].resize(numPts);
  }//end for i

  assert(dao != NULL);

  if(dao->iAmActive()) {
    std::vector<ot::TreeNode> minBlocks = dao->getMinAllBlocks();

    int rankActive = dao->getRankActive();
    int npesActive = dao->getNpesActive();
    MPI_Comm commActive = dao->getCommActive();

    unsigned int maxDepth = dao->getMaxDepth();
    double fac = static_cast<double>((1u << (maxDepth - 1)));

    unsigned int* part = NULL;
    if(!(outOfBndsList.empty())) {
      part = new unsigned int[outOfBndsList.size()]; 
      assert(part);
    }

    int* sendSizes = new int[npesActive];
    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = 0;
      assert(sendSizes);
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      assert(outOfBndsList[l] < xPosArr.size());
      assert(outOfBndsList[l] < yPosArr.size());
      assert(outOfBndsList[l] < zPosArr.size());
      double xPos = xPosArr[outOfBndsList[l]];
      double yPos = yPosArr[outOfBndsList[l]];
      double zPos = zPosArr[outOfBndsList[l]];
      if( (xPos < 0.0) || (xPos >= 1.0) || 
          (yPos < 0.0) || (yPos >= 1.0) || 
          (zPos < 0.0) || (zPos >= 1.0) ) {
        part[l] = rankActive;
      } else {
        unsigned int xint = static_cast<unsigned int>(fac*xPos);
        unsigned int yint = static_cast<unsigned int>(fac*yPos);
        unsigned int zint = static_cast<unsigned int>(fac*zPos);

        ot::TreeNode tmpNode(xint, yint, zint, maxDepth, 3, maxDepth);
        bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks, tmpNode, part[l], NULL, NULL);
        assert(found);
      }
      assert(part[l] < npesActive);
      sendSizes[part[l]] += 3;
    }//end for l

    int* recvSizes = new int[npesActive];
    assert(recvSizes);

    par::Mpi_Alltoall<int>(sendSizes, recvSizes, 1, commActive);

    int* sendOff = new int[npesActive];
    assert(sendOff);

    int* recvOff = new int[npesActive];
    assert(recvOff);

    sendOff[0] = 0;
    recvOff[0] = 0;
    for(int i = 1; i < npesActive; i++) {
      sendOff[i] = sendOff[i - 1] + sendSizes[i - 1];
      recvOff[i] = recvOff[i - 1] + recvSizes[i - 1];
    }//end for i

    double* sendPts = NULL;
    if(!(outOfBndsList.empty())) {
      sendPts = new double[3*(outOfBndsList.size())];
      assert(sendPts);
    }

    double* recvPts = NULL;
    if( recvOff[npesActive - 1] + recvSizes[npesActive - 1] ) {
      recvPts = new double[recvOff[npesActive - 1] + recvSizes[npesActive - 1]];
      assert(recvPts);
    }

    int* tmpSendSizes = new int[npesActive];
    assert(tmpSendSizes);
    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      assert(outOfBndsList[l] < xPosArr.size());
      assert(outOfBndsList[l] < yPosArr.size());
      assert(outOfBndsList[l] < zPosArr.size());
      double xPos = xPosArr[outOfBndsList[l]];
      double yPos = yPosArr[outOfBndsList[l]];
      double zPos = zPosArr[outOfBndsList[l]];
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      sendPts[ptIdx] = xPos;
      sendPts[ptIdx + 1] = yPos;
      sendPts[ptIdx + 2] = zPos;
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += 3;
    }//end for l

    par::Mpi_Alltoallv_sparse<double>(sendPts, sendSizes, sendOff,
        recvPts, recvSizes, recvOff, commActive);

    unsigned int totalRecvPts = (recvOff[npesActive - 1] + recvSizes[npesActive-1])/3;
    double* tmpSendResults = NULL;
    if(totalRecvPts) {
      tmpSendResults = new double[numImages*totalRecvPts];
      assert(tmpSendResults);
    }

    if(totalRecvPts) {
      assert(!(imgPatches.empty()));
      assert(!(mesh.empty()));
      for(int i = 0; i < numImages; i++) {
        assert(!(F[i].empty()));
        assert(!(gradF[i].empty()));
      }//end for i
    }

    for(int j = 0; j < totalRecvPts; j++) {
      double xPos = recvPts[(3*j)];
      double yPos = recvPts[(3*j) + 1];
      double zPos = recvPts[(3*j) + 2];
      for(int i = 0; i < numImages; i++) {
        tmpSendResults[(numImages*j) + i] = newEvalCubicFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos);
      }//end for i
    }//end for j

    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = (numImages*(sendSizes[i]/3));
      recvSizes[i] = (numImages*(recvSizes[i]/3));
      sendOff[i] = (numImages*(sendOff[i]/3));
      recvOff[i] = (numImages*(recvOff[i]/3));
    }//end for i

    double* tmpRecvResults = NULL;
    if(!(outOfBndsList.empty())) {
      tmpRecvResults = new double[numImages*(outOfBndsList.size())];
      assert(tmpRecvResults);
    }

    par::Mpi_Alltoallv_sparse<double>(tmpSendResults, recvSizes, recvOff,
        tmpRecvResults, sendSizes, sendOff, commActive);

    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      for(int i = 0; i < numImages; i++) {
        assert( outOfBndsList[l] < (results[i]).size() );
        results[i][outOfBndsList[l]] = tmpRecvResults[ptIdx + i];
      }//end for i
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += numImages;
    }//end for l

    if(tmpSendResults) {
      delete [] tmpSendResults;
      tmpSendResults = NULL;
    }

    if(tmpRecvResults) {
      delete [] tmpRecvResults;
      tmpRecvResults = NULL;
    }

    if(part) {
      delete [] part;
      part = NULL;
    }

    if(sendPts) {
      delete [] sendPts;
      sendPts = NULL;
    }

    if(recvPts) {
      delete [] recvPts;
      recvPts = NULL;
    }

    assert(tmpSendSizes != NULL);
    delete [] tmpSendSizes;
    tmpSendSizes = NULL;

    assert(sendOff != NULL);
    delete [] sendOff;
    sendOff = NULL;

    assert(recvOff != NULL);
    delete [] recvOff;
    recvOff = NULL;

    assert(sendSizes != NULL);
    delete [] sendSizes;
    sendSizes = NULL;

    assert(recvSizes != NULL);
    delete [] recvSizes;
    recvSizes = NULL;
  }//end if active

  //Process Local Portion
  for(int i = 0; i < numImages; i++) {
    if(!(outOfBndsList.empty())) {
      for(int l = 0; l < outOfBndsList.size(); l++) {
        int jSt = 0;
        if(l) {
          jSt = outOfBndsList[l - 1] + 1;
        }
        for(int j = jSt; j < outOfBndsList[l]; j++) {
          double xPos = xPosArr[j];
          double yPos = yPosArr[j];
          double zPos = zPosArr[j];
          results[i][j] = newEvalCubicFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos);
        }//end for j
      }//end for l
      for(int j = outOfBndsList[outOfBndsList.size() - 1] + 1; j < numPts; j++) {
        double xPos = xPosArr[j];
        double yPos = yPosArr[j];
        double zPos = zPosArr[j];
        results[i][j] = newEvalCubicFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos);
      }//end for j
    } else {
      for(int j = 0; j < numPts; j++) {
        double xPos = xPosArr[j];
        double yPos = yPosArr[j];
        double zPos = zPosArr[j];
        results[i][j] = newEvalCubicFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos);
      }//end for j
    }//end if anything to skip
  }//end for i

}


void newEvalGradFnAtAllPts(ot::DA* dao, const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    const std::vector<unsigned int>& outOfBndsList,
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results) {

  int numImages = F.size();
  int numPts = xPosArr.size();

  unsigned int Ne;
  if(!(imgPatches.empty())) {
    unsigned int regLev = imgPatches[0].getLevel();
    Ne = (1u << (regLev - 1));
  }

  results.resize(numImages);
  for(int i = 0; i < numImages; i++) {
    results[i].resize(3*numPts);
  }//end for i

  assert(dao != NULL);

  if(dao->iAmActive()) {
    std::vector<ot::TreeNode> minBlocks = dao->getMinAllBlocks();

    int rankActive = dao->getRankActive();
    int npesActive = dao->getNpesActive();
    MPI_Comm commActive = dao->getCommActive();

    unsigned int maxDepth = dao->getMaxDepth();
    double fac = static_cast<double>((1u << (maxDepth - 1)));

    unsigned int* part = NULL;
    if(!(outOfBndsList.empty())) {
      part = new unsigned int[outOfBndsList.size()]; 
    }

    int* sendSizes = new int[npesActive];
    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      assert(outOfBndsList[l] < xPosArr.size());
      assert(outOfBndsList[l] < yPosArr.size());
      assert(outOfBndsList[l] < zPosArr.size());
      double xPos = xPosArr[outOfBndsList[l]];
      double yPos = yPosArr[outOfBndsList[l]];
      double zPos = zPosArr[outOfBndsList[l]];
      if( (xPos < 0.0) || (xPos >= 1.0) || 
          (yPos < 0.0) || (yPos >= 1.0) || 
          (zPos < 0.0) || (zPos >= 1.0) ) {
        part[l] = rankActive;
      } else {
        unsigned int xint = static_cast<unsigned int>(fac*xPos);
        unsigned int yint = static_cast<unsigned int>(fac*yPos);
        unsigned int zint = static_cast<unsigned int>(fac*zPos);

        ot::TreeNode tmpNode(xint, yint, zint, maxDepth, 3, maxDepth);
        bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks, tmpNode, part[l], NULL, NULL);
        assert(found);
      }
      assert(part[l] < npesActive);
      sendSizes[part[l]] += 3;
    }//end for l

    int* recvSizes = new int[npesActive];
    assert(recvSizes);

    par::Mpi_Alltoall<int>(sendSizes, recvSizes, 1, commActive);

    int* sendOff = new int[npesActive];
    assert(sendOff);

    int* recvOff = new int[npesActive];
    assert(recvOff);

    sendOff[0] = 0;
    recvOff[0] = 0;
    for(int i = 1; i < npesActive; i++) {
      sendOff[i] = sendOff[i - 1] + sendSizes[i - 1];
      recvOff[i] = recvOff[i - 1] + recvSizes[i - 1];
    }//end for i

    double* sendPts = NULL;
    if(!(outOfBndsList.empty())) {
      sendPts = new double[3*(outOfBndsList.size())];
      assert(sendPts);
    }

    double* recvPts = NULL;
    if( recvOff[npesActive - 1] + recvSizes[npesActive - 1] ) {
      recvPts = new double[recvOff[npesActive - 1] + recvSizes[npesActive - 1]];
      assert(recvPts);
    }

    int* tmpSendSizes = new int[npesActive];
    assert(tmpSendSizes);
    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      assert(outOfBndsList[l] < xPosArr.size());
      assert(outOfBndsList[l] < yPosArr.size());
      assert(outOfBndsList[l] < zPosArr.size());
      double xPos = xPosArr[outOfBndsList[l]];
      double yPos = yPosArr[outOfBndsList[l]];
      double zPos = zPosArr[outOfBndsList[l]];
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      sendPts[ptIdx] = xPos;
      sendPts[ptIdx + 1] = yPos;
      sendPts[ptIdx + 2] = zPos;
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += 3;
    }//end for l

    par::Mpi_Alltoallv_sparse<double>(sendPts, sendSizes, sendOff,
        recvPts, recvSizes, recvOff, commActive);

    unsigned int totalRecvPts = (recvOff[npesActive - 1] + recvSizes[npesActive-1])/3;
    double* tmpSendResults = NULL;
    if(totalRecvPts) {
      tmpSendResults = new double[3*numImages*totalRecvPts];
      assert(tmpSendResults);
    }

    if(totalRecvPts) {
      assert(!(imgPatches.empty()));
    }

    for(int j = 0; j < totalRecvPts; j++) {
      double xPos = recvPts[(3*j)];
      double yPos = recvPts[(3*j) + 1];
      double zPos = recvPts[(3*j) + 2];
      for(int i = 0; i < numImages; i++) {
        double res[3];
        newEvalCubicGradFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos, res);
        for(int k = 0; k < 3; k++) {
          tmpSendResults[(((numImages*j) + i)*3) + k] = res[k]*2.0*static_cast<double>(Ne);
        }//end for k         
      }//end for i
    }//end for j

    for(int i = 0; i < npesActive; i++) {
      sendSizes[i] = (numImages*(sendSizes[i]));
      recvSizes[i] = (numImages*(recvSizes[i]));
      sendOff[i] = (numImages*(sendOff[i]));
      recvOff[i] = (numImages*(recvOff[i]));
    }//end for i

    double* tmpRecvResults = NULL;
    if(!(outOfBndsList.empty())) {
      tmpRecvResults = new double[3*numImages*(outOfBndsList.size())];
      assert(tmpRecvResults);
    }

    par::Mpi_Alltoallv_sparse<double>(tmpSendResults, recvSizes, recvOff,
        tmpRecvResults, sendSizes, sendOff, commActive);

    for(int i = 0; i < npesActive; i++) {
      tmpSendSizes[i] = 0;
    }//end for i

    for(int l = 0; l < outOfBndsList.size(); l++) {
      unsigned int ptIdx = sendOff[part[l]] + tmpSendSizes[part[l]]; 
      for(int i = 0; i < numImages; i++) {
        for(int k = 0; k < 3; k++) {
          assert( ((3*(outOfBndsList[l])) + k) < results[i].size() );
          results[i][(3*(outOfBndsList[l])) + k] = tmpRecvResults[ptIdx + (3*i) + k];
        }//end for k
      }//end for i
      assert(part[l] < npesActive);
      tmpSendSizes[part[l]] += (3*numImages);
    }//end for l

    if(tmpSendResults) {
      delete [] tmpSendResults;
      tmpSendResults = NULL;
    }

    if(tmpRecvResults) {
      delete [] tmpRecvResults;
      tmpRecvResults = NULL;
    }

    if(part) {
      delete [] part;
      part = NULL;
    }

    if(sendPts) {
      delete [] sendPts;
      sendPts = NULL;
    }

    if(recvPts) {
      delete [] recvPts;
      recvPts = NULL;
    }

    assert(tmpSendSizes);
    delete [] tmpSendSizes;
    tmpSendSizes = NULL;

    assert(sendOff);
    delete [] sendOff;
    sendOff = NULL;

    assert(recvOff);
    delete [] recvOff;
    recvOff = NULL;

    assert(sendSizes);
    delete [] sendSizes;
    sendSizes = NULL;

    assert(recvSizes);
    delete [] recvSizes;
    recvSizes = NULL;
  }//end if active


  //Process Local Portion
  for(int i = 0; i < numImages; i++) {
    if(!(outOfBndsList.empty())) {
      for(int l = 0; l < outOfBndsList.size(); l++) {
        int jSt = 0;
        if(l) {
          jSt = outOfBndsList[l - 1] + 1;
        }
        for(int j = jSt; j < outOfBndsList[l]; j++) {
          double xPos = xPosArr[j];
          double yPos = yPosArr[j];
          double zPos = zPosArr[j];
          double res[3];
          newEvalCubicGradFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos, res);
          for(int k = 0; k < 3; k++) {
            results[i][(3*j) + k] = res[k]*2.0*static_cast<double>(Ne);
          }//end for k
        }//end for j
      }//end for l
      for(int j = outOfBndsList[outOfBndsList.size() - 1] + 1; j < numPts; j++) {
        double xPos = xPosArr[j];
        double yPos = yPosArr[j];
        double zPos = zPosArr[j];
        double res[3];
        newEvalCubicGradFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos, res);
        for(int k = 0; k < 3; k++) {
          results[i][(3*j) + k] = res[k]*2.0*static_cast<double>(Ne);
        }//end for k
      }//end for j
    } else {
      for(int j = 0; j < numPts; j++) {
        double xPos = xPosArr[j];
        double yPos = yPosArr[j];
        double zPos = zPosArr[j];
        double res[3];
        newEvalCubicGradFn(F[i], gradF[i], imgPatches, mesh, xPos, yPos, zPos, res);
        for(int k = 0; k < 3; k++) {
          results[i][(3*j) + k] = res[k]*2.0*static_cast<double>(Ne);
        }//end for k
      }//end for j
    }//end if anything to skip
  }//end for i
}


double newEvalCubicFn(const std::vector<double>& fArr, 
    const std::vector<double>& gArr, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, double xPos, double yPos, double zPos) {

  double res;

  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    res = 0.0;
  } else {   

    assert(!(imgPatches.empty()));
    unsigned int maxDepth = imgPatches[0].getMaxDepth();
    unsigned int regLev = imgPatches[0].getLevel();
    unsigned int Ne = (1u << (regLev - 1));
    double h = 1.0/(static_cast<double>(Ne));

    double hFac = (1u << (maxDepth - 1));
    unsigned int xint = static_cast<unsigned int>(xPos*hFac);
    unsigned int yint = static_cast<unsigned int>(yPos*hFac);
    unsigned int zint = static_cast<unsigned int>(zPos*hFac);
    ot::TreeNode searchNode(xint, yint, zint, maxDepth, 3, maxDepth);

    unsigned int idx;
    bool found = seq::maxLowerBound<ot::TreeNode>(imgPatches, searchNode, idx, NULL, NULL);

    assert(idx < imgPatches.size());
    assert(found);
    assert( (imgPatches[idx].isAncestor(searchNode)) || (imgPatches[idx] == searchNode) );

    double x0 = (static_cast<double>(imgPatches[idx].getX()))/hFac;
    double y0 = (static_cast<double>(imgPatches[idx].getY()))/hFac;
    double z0 = (static_cast<double>(imgPatches[idx].getZ()))/hFac;

    double psi = ((xPos - x0)*2.0/h) - 1.0;
    double eta = ((yPos - y0)*2.0/h) - 1.0;
    double gamma = ((zPos - z0)*2.0/h) - 1.0;

    assert(psi >= -1);
    assert(psi <= 1);
    assert(eta >= -1);
    assert(eta <= 1);
    assert(gamma >= -1);
    assert(gamma <= 1);

    double phiVals[8][4];
    evalAll3Dcubic(psi, eta, gamma, phiVals);

    assert(fArr.size() == imgPatches.size());

    res = 0.0;
    for(int i = 0; i < 8; i++) {
      assert( ((8*idx) + i) < mesh.size() );
      assert( mesh[(8*idx) + i] < fArr.size() );
      assert( ((3*mesh[(8*idx) + i]) + 2) < gArr.size() );
      res += (fArr[mesh[(8*idx) + i]]*phiVals[i][0]);
      for(int j = 0; j < 3; j++) {
        res += (0.5*h*gArr[(3*(mesh[(8*idx) + i])) + j]*phiVals[i][1 + j]);
      }
    }
  }

  return res;
}


void newEvalCubicGradFn(const std::vector<double>& fArr, 
    const std::vector<double>& gArr, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, double xPos, double yPos, double zPos, double* res) {


  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    res[0] = 0.0;
    res[1] = 0.0;
    res[2] = 0.0;
  } else {   

    assert(!(imgPatches.empty()));
    unsigned int maxDepth = imgPatches[0].getMaxDepth();
    unsigned int regLev = imgPatches[0].getLevel();
    unsigned int Ne = (1u << (regLev - 1));
    double h = 1.0/(static_cast<double>(Ne));

    double hFac = (1u << (maxDepth - 1));

    unsigned int xint = static_cast<unsigned int>(xPos*hFac);
    unsigned int yint = static_cast<unsigned int>(yPos*hFac);
    unsigned int zint = static_cast<unsigned int>(zPos*hFac);
    ot::TreeNode searchNode(xint, yint, zint, maxDepth, 3, maxDepth);

    unsigned int idx;
    bool found = seq::maxLowerBound<ot::TreeNode>(imgPatches, searchNode, idx, NULL, NULL);

    assert(found);
    assert( idx < imgPatches.size() );
    assert( (imgPatches[idx].isAncestor(searchNode)) || (imgPatches[idx] == searchNode) );

    double x0 = (static_cast<double>(imgPatches[idx].getX()))/hFac;
    double y0 = (static_cast<double>(imgPatches[idx].getY()))/hFac;
    double z0 = (static_cast<double>(imgPatches[idx].getZ()))/hFac;

    double psi = ((xPos - x0)*2.0/h) - 1.0;
    double eta = ((yPos - y0)*2.0/h) - 1.0;
    double gamma = ((zPos - z0)*2.0/h) - 1.0;

    assert(psi >= -1);
    assert(psi <= 1);
    assert(eta >= -1);
    assert(eta <= 1);
    assert(gamma >= -1);
    assert(gamma <= 1);

    double gradPhiVals[8][4][3];
    evalAll3DcubicGrad(psi, eta, gamma, gradPhiVals);

    res[0] = 0.0;
    res[1] = 0.0;
    res[2] = 0.0;
    for(int i = 0; i < 8; i++) {
      assert( ((8*idx) + i) < mesh.size() );
      assert(mesh[(8*idx) + i] < fArr.size());
      assert( ((3*mesh[(8*idx) + i]) + 2) < gArr.size() );
      for(int j = 0; j < 3; j++) {
        res[j] += (fArr[mesh[(8*idx) + i]]*gradPhiVals[i][0][j]);
        for(int k = 0; k < 3; k++) {
          res[j] += (0.5*h*gArr[(3*(mesh[(8*idx) + i])) + k]*gradPhiVals[i][1 + k][j]);
        }//end for k
      }//end for j
    }//end for i
  }
}


void computeNodalTauAtU(ot::DA* dao, int Ne,
    const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalTauAtU) {
  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  composeXposAtUnodal(dao, U, xPosArr, yPosArr, zPosArr);

  //evaluate at all points
  evalFnAtAllPts(xPosArr, yPosArr, zPosArr, Ne, tau, gradTau, nodalTauAtU);
}


void computeNodalGradTauAtU(ot::DA* dao, int Ne, const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalGradTauAtU) {
  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  composeXposAtUnodal(dao, U, xPosArr, yPosArr, zPosArr);

  //evaluate at all points
  evalGradFnAtAllPts(xPosArr, yPosArr, zPosArr, Ne, tau, gradTau, nodalGradTauAtU);
}


void computeSigVals(ot::DA* dao, int Ne, const std::vector<std::vector<double> >& sig,
    const std::vector<std::vector<double> >& gradSig, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals) {

  //Assumption: The entire image is replicated on all processors 

  int numImages = sig.size();

  sigVals.resize(numImages);

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
            double xPos = (x0 + (0.5*hOct*(1.0 + gPts[m])));
            double yPos = (y0 + (0.5*hOct*(1.0 + gPts[n])));
            double zPos = (z0 + (0.5*hOct*(1.0 + gPts[p])));
            for(int i = 0; i < numImages; i++) {
              const double* fArr = &(*(sig[i].begin()));
              const double* gArr = &(*(gradSig[i].begin()));
              sigVals[i].push_back(evalCubicFn(fArr, gArr, Ne, xPos, yPos, zPos));
            }
          }
        }
      }
    }
  }
}


void computeGradTauAtU(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU) { 

  //Assumption: The entire image is replicated on all processors 

  int numImages = tau.size();

  gradTauAtU.resize(numImages);

  typedef double* doublePtr;
  typedef double** double2Ptr;

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
            double xPos = (x0 + (0.5*hOct*(1.0 + gPts[m])) + u1loc[m][n][p]);
            double yPos = (y0 + (0.5*hOct*(1.0 + gPts[n])) + u2loc[m][n][p]);
            double zPos = (z0 + (0.5*hOct*(1.0 + gPts[p])) + u3loc[m][n][p]);
            for(int i = 0; i < numImages; i++) {
              const double* fArr = &(*(tau[i].begin()));
              const double* gArr = &(*(gradTau[i].begin()));
              double res[3];
              evalCubicGradFn(fArr, gArr, Ne, xPos, yPos, zPos, res);
              for(int k = 0; k < 3; k++) {
                gradTauAtU[i].push_back(res[k]*2.0*static_cast<double>(Ne));
              }
            }
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


void computeTauAtU(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU) {

  //Assumption: The entire image is replicated on all processors 

  int numImages = tau.size();

  tauAtU.resize(numImages);

  typedef double* doublePtr;
  typedef double** double2Ptr;

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
            double xPos = (x0 + (0.5*hOct*(1.0 + gPts[m])) + u1loc[m][n][p]);
            double yPos = (y0 + (0.5*hOct*(1.0 + gPts[n])) + u2loc[m][n][p]);
            double zPos = (z0 + (0.5*hOct*(1.0 + gPts[p])) + u3loc[m][n][p]);
            for(int i = 0; i < numImages; i++) {
              const double* fArr = &(*(tau[i].begin()));
              const double* gArr = &(*(gradTau[i].begin()));
              tauAtU[i].push_back(evalCubicFn(fArr, gArr, Ne, xPos, yPos, zPos));
            }
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


void computeRegularNodalTauAtU(int Ne, const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalTauAtU) {
  std::vector<double> xPosArr((Ne + 1)*(Ne + 1)*(Ne + 1));
  std::vector<double> yPosArr((Ne + 1)*(Ne + 1)*(Ne + 1));
  std::vector<double> zPosArr((Ne + 1)*(Ne + 1)*(Ne + 1));

  PetscScalar* uArr;
  VecGetArray(U, &uArr);

  double h = 1.0/static_cast<double>(Ne);

  int cnt = 0;
  for(int k = 0; k < (Ne + 1); k++) {
    for(int j = 0; j < (Ne + 1); j++) {
      for(int i = 0; i < (Ne + 1); i++) {
        xPosArr[cnt] = (static_cast<double>(i)*h) + uArr[3*cnt];
        yPosArr[cnt] = (static_cast<double>(j)*h) + uArr[(3*cnt) + 1];
        zPosArr[cnt] = (static_cast<double>(k)*h) + uArr[(3*cnt) + 2];
        cnt++;
      }
    }
  }

  VecRestoreArray(U, &uArr);

  evalFnAtAllPts(xPosArr, yPosArr, zPosArr, Ne, tau, gradTau, nodalTauAtU);
}


void evalFnAtAllPts(const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    int Ne, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results) {

  int numImages = F.size();
  int numPts = xPosArr.size();

  results.resize(numImages);
  for(int i = 0; i < numImages; i++) {
    results[i].resize(numPts);
    const double* fArr = &(*(F[i].begin()));
    const double* gArr = &(*(gradF[i].begin()));
    for(int j = 0; j < numPts; j++) {
      double xPos = xPosArr[j];
      double yPos = yPosArr[j];
      double zPos = zPosArr[j];
      results[i][j] = evalCubicFn(fArr, gArr, Ne, xPos, yPos, zPos);
    }
  }
}


void evalGradFnAtAllPts(const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    int Ne, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results) {

  int numImages = F.size();
  int numPts = xPosArr.size();

  results.resize(numImages);
  for(int i = 0; i < numImages; i++) {
    results[i].resize(3*numPts);
    const double* fArr = &(*(F[i].begin()));
    const double* gArr = &(*(gradF[i].begin()));
    for(int j = 0; j < numPts; j++) {
      double xPos = xPosArr[j];
      double yPos = yPosArr[j];
      double zPos = zPosArr[j];
      double res[3];
      evalCubicGradFn(fArr, gArr, Ne, xPos, yPos, zPos, res);
      for(int k = 0; k < 3; k++) {
        results[i][(3*j) + k] = res[k]*2.0*static_cast<double>(Ne);
      }
    }
  }
}


double evalCubicFn(const double* fArr, const double* gArr, 
    int Ne, double xPos, double yPos, double zPos) {

  double h = 1.0/static_cast<double>(Ne); 
  double res;

  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    res = 0.0;
  } else {   
    int ei = static_cast<int>(xPos/h);
    int ej = static_cast<int>(yPos/h);
    int ek = static_cast<int>(zPos/h);
    double x0 = static_cast<double>(ei)*h;
    double y0 = static_cast<double>(ej)*h;
    double z0 = static_cast<double>(ek)*h;
    double psi = ((xPos - x0)*2.0/h) - 1.0;
    double eta = ((yPos - y0)*2.0/h) - 1.0;
    double gamma = ((zPos - z0)*2.0/h) - 1.0;
    double phiVals[8][4];

    evalAll3Dcubic(psi, eta, gamma, phiVals);

    res = 0.0;
    for(int i = 0; i < 8; i++) {
      int xid = ei + (i%2);
      int yid = ej + ((i/2)%2);
      int zid = ek + (i/4);
      res += (fArr[(((zid*(Ne + 1)) + yid)*(Ne + 1)) + xid]*phiVals[i][0]);
      for(int j = 0; j < 3; j++) {
        res += (0.5*h*gArr[(((((zid*(Ne + 1)) + yid)*(Ne + 1)) + xid)*3) + j]*phiVals[i][1 + j]);
      }
    }
  }

  return res;
}


void evalCubicGradFn(const double* fArr, const double* gArr, 
    int Ne, double xPos, double yPos, double zPos, double* res) {

  double h = 1.0/static_cast<double>(Ne); 
  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    res[0] = 0.0;
    res[1] = 0.0;
    res[2] = 0.0;
  } else {   
    int ei = static_cast<int>(xPos/h);
    int ej = static_cast<int>(yPos/h);
    int ek = static_cast<int>(zPos/h);
    double x0 = static_cast<double>(ei)*h;
    double y0 = static_cast<double>(ej)*h;
    double z0 = static_cast<double>(ek)*h;
    double psi = ((xPos - x0)*2.0/h) - 1.0;
    double eta = ((yPos - y0)*2.0/h) - 1.0;
    double gamma = ((zPos - z0)*2.0/h) - 1.0;
    double gradPhiVals[8][4][3];

    evalAll3DcubicGrad(psi, eta, gamma, gradPhiVals);

    res[0] = 0.0;
    res[1] = 0.0;
    res[2] = 0.0;
    for(int i = 0; i < 8; i++) {
      int xid = ei + (i%2);
      int yid = ej + ((i/2)%2);
      int zid = ek + (i/4);
      for(int j = 0; j < 3; j++) {
        res[j] += (fArr[(((zid*(Ne + 1)) + yid)*(Ne + 1)) + xid]*gradPhiVals[i][0][j]);
        for(int k = 0; k < 3; k++) {
          res[j] += (0.5*h*gArr[(((((zid*(Ne + 1)) + yid)*(Ne + 1)) + xid)*3) + k]*gradPhiVals[i][1 + k][j]);
        }
      }
    }
  }
}


void evalAll3Dcubic(double psi, double eta, double gamma, double phiArr[8][4]) {
  int psiMap[] = {0, 1, 0, 1, 0, 1, 0, 1};
  int etaMap[] = {0, 0, 1, 1, 0, 0, 1, 1};
  int gammaMap[] = {0, 0, 0, 0, 1, 1, 1, 1};
  for(int i = 0; i < 8; i++) {
    phiArr[i][0] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));

    phiArr[i][1] = (eval1Dcubic(psiMap[i], 1, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));

    phiArr[i][2] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 1, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));

    phiArr[i][3] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 1, gamma));
  }
}


void evalAll3DcubicGrad(double psi, double eta, double gamma, double gradPhiArr[8][4][3]) {
  int psiMap[] = {0, 1, 0, 1, 0, 1, 0, 1};
  int etaMap[] = {0, 0, 1, 1, 0, 0, 1, 1};
  int gammaMap[] = {0, 0, 0, 0, 1, 1, 1, 1};
  for(int i = 0; i < 8; i++) {
    gradPhiArr[i][0][0] = (eval1DcubicGrad(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));
    gradPhiArr[i][0][1] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1DcubicGrad(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));
    gradPhiArr[i][0][2] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1DcubicGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][1][0] = (eval1DcubicGrad(psiMap[i], 1, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));
    gradPhiArr[i][1][1] = (eval1Dcubic(psiMap[i], 1, psi)*
        eval1DcubicGrad(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));
    gradPhiArr[i][1][2] = (eval1Dcubic(psiMap[i], 1, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1DcubicGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][2][0] = (eval1DcubicGrad(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 1, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));
    gradPhiArr[i][2][1] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1DcubicGrad(etaMap[i], 1, eta)*
        eval1Dcubic(gammaMap[i], 0, gamma));
    gradPhiArr[i][2][2] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 1, eta)*
        eval1DcubicGrad(gammaMap[i], 0, gamma));

    gradPhiArr[i][3][0] = (eval1DcubicGrad(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 1, gamma));
    gradPhiArr[i][3][1] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1DcubicGrad(etaMap[i], 0, eta)*
        eval1Dcubic(gammaMap[i], 1, gamma));
    gradPhiArr[i][3][2] = (eval1Dcubic(psiMap[i], 0, psi)*
        eval1Dcubic(etaMap[i], 0, eta)*
        eval1DcubicGrad(gammaMap[i], 1, gamma));
  }
}


double eval1Dcubic(int nodeNum, int compNum, double psi)
{
  double phi;

  switch(nodeNum) {
    case 0: {
              switch(compNum) {
                case 0: {
                          phi = 0.5  - (0.75*psi) + (0.25*CUBE(psi));
                          break;
                        }
                case 1: {
                          phi = 0.25 - (0.25*psi) - (0.25*SQR(psi)) + (0.25*CUBE(psi));
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
                          phi = 0.5 + (0.75*psi) - (0.25*CUBE(psi));
                          break;
                        }
                case 1: {
                          phi = -0.25 - (0.25*psi) + (0.25*SQR(psi)) + (0.25*CUBE(psi));
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


double eval1DcubicGrad(int nodeNum, int compNum, double psi)
{
  double phi;

  switch(nodeNum) {
    case 0: {
              switch(compNum) {
                case 0: {
                          phi =  -0.75 + (0.75*SQR(psi));
                          break;
                        }
                case 1: {
                          phi = -0.25 - (0.5*psi) + (0.75*SQR(psi));
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
                          phi = 0.75 - (0.75*SQR(psi));
                          break;
                        }
                case 1: {
                          phi =  -0.25 + (0.5*psi) + (0.75*SQR(psi));
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







