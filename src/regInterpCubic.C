
/**
  @file registration.C
  @brief Components of Elastic Registration
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  */

#include "mpi.h"
#include "oct/TreeNode.h"
#include <vector>
#include "seq/seqUtils.h"
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

void computeNodalGradTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal,
    const std::vector<std::vector<double> > & gradTauLocal, int Ne, int padding, int numImages,
    Vec U, std::vector<double >& nodalGradTauAtU) {

  PetscLogEventBegin(computeNodalGradTauEvent, 0, 0, 0, 0);

  assert(dao != NULL);

  unsigned int maxD;
  unsigned int balOctMaxD;
  double hFac;
  double patchWidth;
  std::vector<double> xyzVec;
  std::vector<char> markedNode;

  if(dao->iAmActive()) {
    maxD = dao->getMaxDepth();
    balOctMaxD = maxD - 1;
    hFac = 1.0/static_cast<double>(1u << balOctMaxD);
    patchWidth = static_cast<double>(padding)/static_cast<double>(Ne);

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
  }//end if active

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;
  std::vector<unsigned int> outOfBndsList;

  unsigned int numPts = (xyzVec.size())/3;

  std::vector<ot::TreeNode> blocks = dao->getBlocks();

  nodalGradTauAtU.resize(numPts*numImages*3);

  for(int i = 0; i < numPts; i++) {
    double xPos = xyzVec[3*i];
    double yPos = xyzVec[(3*i) + 1];
    double zPos = xyzVec[(3*i) + 2];
    if(markedNode[i]) {
      outOfBndsList.push_back(i);
      xPosArr.push_back(xPos);
      yPosArr.push_back(yPos);
      zPosArr.push_back(zPos);
    } else {
      std::vector<double> tmpVals;
      evalCubicGradFn(blocks, tauLocal, gradTauLocal, Ne, padding, numImages, 
          xPos, yPos, zPos, tmpVals);
      for(int j = 0; j < (3*numImages); j++) {
        nodalGradTauAtU[(i*numImages*3) + j]  = tmpVals[j];
      }//end for j
    }
  }//end for i

  //Remote Portion
  std::vector<double> tmpResults;

  if(dao->iAmActive()) {
    evalGradFnAtAllPts(dao, tauLocal, gradTauLocal, Ne, padding, numImages,
        xPosArr, yPosArr, zPosArr, tmpResults);
  }

  for(int i = 0; i < outOfBndsList.size(); i++) {
    for(int j = 0; j < (3*numImages); j++) {
      nodalGradTauAtU[(outOfBndsList[i]*numImages*3) + j]  = tmpResults[(i*numImages*3) + j];
    }//end for j
  }//end for i

  PetscLogEventEnd(computeNodalGradTauEvent, 0, 0, 0, 0);

}

void computeNodalTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal,
    const std::vector<std::vector<double> > & gradTauLocal, int Ne, int padding, int numImages,
    Vec U, std::vector<double >& nodalTauAtU) {

  PetscLogEventBegin(computeNodalTauEvent, 0, 0, 0, 0);

  assert(dao != NULL);
  assert(dao->iAmActive());

  unsigned int maxD = dao->getMaxDepth();
  unsigned int balOctMaxD = maxD - 1;
  double hFac = 1.0/static_cast<double>(1u << balOctMaxD);
  double patchWidth = static_cast<double>(padding)/static_cast<double>(Ne);

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

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;
  std::vector<unsigned int> outOfBndsList;

  unsigned int numPts = (xyzVec.size())/3;

  std::vector<ot::TreeNode> blocks = dao->getBlocks();

  nodalTauAtU.resize(numPts*numImages);

  for(int i = 0; i < numPts; i++) {
    double xPos = xyzVec[3*i];
    double yPos = xyzVec[(3*i) + 1];
    double zPos = xyzVec[(3*i) + 2];
    if(markedNode[i]) {
      outOfBndsList.push_back(i);
      xPosArr.push_back(xPos);
      yPosArr.push_back(yPos);
      zPosArr.push_back(zPos);
    } else {
      std::vector<double> tmpVals;
      evalCubicFn(blocks, tauLocal, gradTauLocal, Ne, padding, numImages, 
          xPos, yPos, zPos, tmpVals);
      for(int j = 0; j < numImages; j++) {
        nodalTauAtU[(i*numImages) + j]  = tmpVals[j];
      }//end for j
    }
  }//end for i

  //Remote Portion
  std::vector<double> tmpResults;

  evalFnAtAllPts(dao, tauLocal, gradTauLocal, Ne, padding, numImages,
      xPosArr, yPosArr, zPosArr, tmpResults);

  for(int i = 0; i < outOfBndsList.size(); i++) {
    for(int j = 0; j < numImages; j++) {
      nodalTauAtU[(outOfBndsList[i]*numImages) + j]  = tmpResults[(i*numImages) + j];
    }//end for j
  }//end for i

  PetscLogEventEnd(computeNodalTauEvent, 0, 0, 0, 0);

}

void computeSigVals(ot::DA* dao, const std::vector<std::vector<double> > & sigLocal, 
    const std::vector<std::vector<double> > & gradSigLocal,
    int Ne, int padding, int numImages, int numGpts, double* gPts,
    std::vector<double> & sigVals) {

  PetscLogEventBegin(computeSigEvent, 0, 0, 0, 0);

  assert(dao != NULL);

  unsigned int elemSize = dao->getElementSize();
  unsigned int maxDepth;
  double hFac;

  if(dao->iAmActive()) {
    maxDepth = dao->getMaxDepth();
    hFac = 1.0/static_cast<double>(1u << (maxDepth - 1));
  } else {
    assert(elemSize == 0);
  }

  sigVals.resize(numImages*numGpts*numGpts*numGpts*elemSize);

  std::vector<ot::TreeNode> blocks = dao->getBlocks();

  unsigned int elemCtr = 0;
  if(dao->iAmActive()) {
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
            std::vector<double> tmpVals;
            evalCubicFn(blocks, sigLocal, gradSigLocal, Ne, padding, numImages, 
                xPos, yPos, zPos, tmpVals);
            for(int i = 0; i < numImages; i++) {
              sigVals[(((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p)*numImages) + i]  = tmpVals[i];
            }//end i
          }//end for p
        }//end for n
      }//end for m
      elemCtr++;
    }//end WRITABLE
  }//end if active

  assert(elemCtr == elemSize);

  PetscLogEventEnd(computeSigEvent, 0, 0, 0, 0);

}

void computeGradTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal, 
    const std::vector<std::vector<double> > & gradTauLocal, int Ne, int padding,
    int numImages, Vec U, double****** PhiMatStencil, int numGpts,
    double* gPts, std::vector<double> & gradTauAtU) {

  PetscLogEventBegin(computeGradTauEvent, 0, 0, 0, 0);

  assert(dao != NULL);

  std::vector<unsigned int> outOfBndsList;
  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  unsigned int elemSize = dao->getElementSize();

  gradTauAtU.resize(3*numImages*numGpts*numGpts*numGpts*elemSize);

  std::vector<double> tmpResults;

  if(dao->iAmActive()) {
    typedef double* doublePtr;
    typedef double** double2Ptr;

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

    double patchWidth = static_cast<double>(padding)/static_cast<double>(Ne);

    std::vector<ot::TreeNode> blocks = dao->getBlocks();

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
              std::vector<double> tmpVals;
              evalCubicGradFn(blocks, tauLocal, gradTauLocal, Ne, padding, numImages, 
                  xPos, yPos, zPos, tmpVals);
              for(int i = 0; i < (3*numImages); i++) {
                gradTauAtU[(((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p)*numImages*3) + i]  = tmpVals[i];
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
    evalGradFnAtAllPts(dao, tauLocal, gradTauLocal, Ne, padding, numImages,
        xPosArr, yPosArr, zPosArr, tmpResults);

  }//end if active

  for(int i = 0; i < outOfBndsList.size(); i++) {
    for(int j = 0; j < (3*numImages); j++) {
      gradTauAtU[(outOfBndsList[i]*numImages*3) + j]  = tmpResults[(i*numImages*3) + j];
    }//end for j
  }//end for i

  PetscLogEventEnd(computeGradTauEvent, 0, 0, 0, 0);

}


void computeTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal, 
    const std::vector<std::vector<double> > & gradTauLocal, int Ne, int padding,
    int numImages, Vec U, double****** PhiMatStencil, int numGpts,
    double* gPts, std::vector<double> & tauAtU) {

  PetscLogEventBegin(computeTauEvent, 0, 0, 0, 0);

  assert(dao != NULL);

  std::vector<unsigned int> outOfBndsList;
  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  unsigned int elemSize = dao->getElementSize();

  tauAtU.resize(numImages*numGpts*numGpts*numGpts*elemSize);

  std::vector<double> tmpResults;

  if(dao->iAmActive()) {
    typedef double* doublePtr;
    typedef double** double2Ptr;

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

    double patchWidth = static_cast<double>(padding)/static_cast<double>(Ne);

    std::vector<ot::TreeNode> blocks = dao->getBlocks();

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
              std::vector<double> tmpVals;
              evalCubicFn(blocks, tauLocal, gradTauLocal, Ne, padding, numImages, 
                  xPos, yPos, zPos, tmpVals);
              for(int i = 0; i < numImages; i++) {
                tauAtU[(((((((elemCtr*numGpts) + m)*numGpts) + n)*numGpts) + p)*numImages) + i]  = tmpVals[i];
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
    evalFnAtAllPts(dao, tauLocal, gradTauLocal, Ne, padding, numImages,
        xPosArr, yPosArr, zPosArr, tmpResults);

  }//end if active

  for(int i = 0; i < outOfBndsList.size(); i++) {
    for(int j = 0; j < numImages; j++) {
      tauAtU[(outOfBndsList[i]*numImages) + j]  = tmpResults[(i*numImages) + j];
    }//end for j
  }//end for i

  PetscLogEventEnd(computeTauEvent, 0, 0, 0, 0);

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


void evalGradFnAtAllPts(ot::DA* dao, const std::vector<std::vector<double> > &fLocal,
    const std::vector<std::vector<double> > &gLocal,
    int Ne, int padding, int numImages,
    const std::vector<double> & xPosArr, const std::vector<double> & yPosArr,
    const std::vector<double> & zPosArr, std::vector<double> & results) {

  assert(dao != NULL);
  assert(dao->iAmActive());

  unsigned int numPts = xPosArr.size();
  assert(yPosArr.size() == numPts);
  assert(zPosArr.size() == numPts);

  results.resize(3*numImages*numPts);

  std::vector<ot::TreeNode> minBlocks = dao->getMinAllBlocks();

  int rankActive = dao->getRankActive();
  int npesActive = dao->getNpesActive();
  MPI_Comm commActive = dao->getCommActive();

  unsigned int maxDepth = dao->getMaxDepth();

  double hFac = static_cast<double>(1u << (maxDepth - 1));

  unsigned int* part = new unsigned int[numPts];
  assert(part);

  int* sendSizes = new int[npesActive];
  assert(sendSizes);
  for(int i = 0; i < npesActive; i++) {
    sendSizes[i] = 0;
  }//end for i

  for(int i = 0; i < numPts; i++) {
    if( (xPosArr[i] < 0.0) || (xPosArr[i] >= 1.0) ||
        (yPosArr[i] < 0.0) || (yPosArr[i] >= 1.0) ||
        (zPosArr[i] < 0.0) || (zPosArr[i] >= 1.0) ) {
      part[i] = rankActive;
    } else {
      unsigned int xint = static_cast<unsigned int>(hFac*(xPosArr[i]));
      unsigned int yint = static_cast<unsigned int>(hFac*(yPosArr[i]));
      unsigned int zint = static_cast<unsigned int>(hFac*(zPosArr[i]));
      ot::TreeNode key(xint, yint, zint, maxDepth, 3, maxDepth);
      bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks, key, part[i], NULL, NULL);
      assert(found);
    }
    assert(part[i] < npesActive);
    sendSizes[part[i]] += 3;
  }//end for i

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

  double* sendPts = new double[3*numPts];
  assert(sendPts);

  double* recvPts = new double[recvOff[npesActive - 1] + recvSizes[npesActive - 1]];
  assert(recvPts);

  int* tmpSendSizes = new int[npesActive];
  assert(tmpSendSizes);
  for(int i = 0; i < npesActive; i++) {
    tmpSendSizes[i] = 0;
  }//end for i

  for(int i = 0; i < numPts; i++) {
    unsigned int ptIdx = sendOff[part[i]] + tmpSendSizes[part[i]];
    sendPts[ptIdx] = xPosArr[i];
    sendPts[ptIdx + 1] = yPosArr[i];
    sendPts[ptIdx + 2] = zPosArr[i];
    tmpSendSizes[part[i]] += 3;
  }//end for i

  par::Mpi_Alltoallv_sparse<double>(sendPts, sendSizes, sendOff,
      recvPts, recvSizes, recvOff, commActive);

  unsigned int totalRecvPts = (recvOff[npesActive - 1] + recvSizes[npesActive-1])/3;

  double* tmpSendResults = new double[3*numImages*totalRecvPts];
  assert(tmpSendResults);

  std::vector<ot::TreeNode> blocks = dao->getBlocks();

  for(int j = 0; j < totalRecvPts; j++) {
    double xPos = recvPts[(3*j)];
    double yPos = recvPts[(3*j) + 1];
    double zPos = recvPts[(3*j) + 2];

    std::vector<double> tmpResults;
    evalCubicGradFn(blocks, fLocal, gLocal, Ne, padding, numImages, 
        xPos, yPos, zPos, tmpResults);

    for(int i = 0; i < (3*numImages); i++) {
      tmpSendResults[(3*numImages*j) + i] = tmpResults[i];          
    }//end for i
  }//end for j

  for(int i = 0; i < npesActive; i++) {
    sendSizes[i] = (numImages*sendSizes[i]);
    recvSizes[i] = (numImages*recvSizes[i]);
    sendOff[i] = (numImages*sendOff[i]);
    recvOff[i] = (numImages*recvOff[i]);
  }//end for i

  double* tmpRecvResults = new double[3*numImages*numPts];
  assert(tmpRecvResults);

  par::Mpi_Alltoallv_sparse<double>(tmpSendResults, recvSizes, recvOff,
      tmpRecvResults, sendSizes, sendOff, commActive);

  for(int i = 0; i < npesActive; i++) {
    tmpSendSizes[i] = 0;
  }//end for i

  for(int i = 0; i < numPts; i++) {
    unsigned int ptIdx = sendOff[part[i]] + tmpSendSizes[part[i]]; 
    for(int j = 0; j < (3*numImages); j++) {
      results[(3*numImages*i) + j] = tmpRecvResults[ptIdx + j];
    }//end for j
    tmpSendSizes[part[i]] += (3*numImages);
  }//end for i

  delete [] tmpRecvResults;
  tmpRecvResults = NULL;

  delete [] tmpSendResults;
  tmpSendResults = NULL;

  delete [] tmpSendSizes;
  tmpSendSizes = NULL;

  delete [] part;
  part = NULL;

  delete [] sendSizes;
  sendSizes = NULL;

  delete [] recvSizes;
  recvSizes = NULL;

  delete [] sendOff;
  sendOff = NULL;

  delete [] recvOff;
  recvOff = NULL;

  delete [] sendPts;
  sendPts = NULL;

  delete [] recvPts;
  recvPts = NULL;

}


void evalFnAtAllPts(ot::DA* dao, const std::vector<std::vector<double> > &fLocal,
    const std::vector<std::vector<double> > &gLocal,
    int Ne, int padding, int numImages,
    const std::vector<double> & xPosArr, const std::vector<double> & yPosArr,
    const std::vector<double> & zPosArr, std::vector<double> & results) {

  assert(dao != NULL);
  assert(dao->iAmActive());

  unsigned int numPts = xPosArr.size();
  assert(yPosArr.size() == numPts);
  assert(zPosArr.size() == numPts);

  results.resize(numImages*numPts);

  std::vector<ot::TreeNode> minBlocks = dao->getMinAllBlocks();

  int rankActive = dao->getRankActive();
  int npesActive = dao->getNpesActive();
  MPI_Comm commActive = dao->getCommActive();

  unsigned int maxDepth = dao->getMaxDepth();

  double hFac = static_cast<double>(1u << (maxDepth - 1));

  unsigned int* part = new unsigned int[numPts];
  assert(part);

  int* sendSizes = new int[npesActive];
  assert(sendSizes);
  for(int i = 0; i < npesActive; i++) {
    sendSizes[i] = 0;
  }//end for i

  for(int i = 0; i < numPts; i++) {
    if( (xPosArr[i] < 0.0) || (xPosArr[i] >= 1.0) ||
        (yPosArr[i] < 0.0) || (yPosArr[i] >= 1.0) ||
        (zPosArr[i] < 0.0) || (zPosArr[i] >= 1.0) ) {
      part[i] = rankActive;
    } else {
      unsigned int xint = static_cast<unsigned int>(hFac*(xPosArr[i]));
      unsigned int yint = static_cast<unsigned int>(hFac*(yPosArr[i]));
      unsigned int zint = static_cast<unsigned int>(hFac*(zPosArr[i]));
      ot::TreeNode key(xint, yint, zint, maxDepth, 3, maxDepth);
      bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks, key, part[i], NULL, NULL);
      assert(found);
    }
    assert(part[i] < npesActive);
    sendSizes[part[i]] += 3;
  }//end for i

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

  double* sendPts = new double[3*numPts];
  assert(sendPts);

  double* recvPts = new double[recvOff[npesActive - 1] + recvSizes[npesActive - 1]];
  assert(recvPts);

  int* tmpSendSizes = new int[npesActive];
  assert(tmpSendSizes);
  for(int i = 0; i < npesActive; i++) {
    tmpSendSizes[i] = 0;
  }//end for i

  for(int i = 0; i < numPts; i++) {
    unsigned int ptIdx = sendOff[part[i]] + tmpSendSizes[part[i]];
    sendPts[ptIdx] = xPosArr[i];
    sendPts[ptIdx + 1] = yPosArr[i];
    sendPts[ptIdx + 2] = zPosArr[i];
    tmpSendSizes[part[i]] += 3;
  }//end for i

  par::Mpi_Alltoallv_sparse<double>(sendPts, sendSizes, sendOff,
      recvPts, recvSizes, recvOff, commActive);

  unsigned int totalRecvPts = (recvOff[npesActive - 1] + recvSizes[npesActive-1])/3;
  double* tmpSendResults = new double[numImages*totalRecvPts];
  assert(tmpSendResults);

  std::vector<ot::TreeNode> blocks = dao->getBlocks();

  for(int j = 0; j < totalRecvPts; j++) {
    double xPos = recvPts[(3*j)];
    double yPos = recvPts[(3*j) + 1];
    double zPos = recvPts[(3*j) + 2];

    std::vector<double> tmpResults;
    evalCubicFn(blocks, fLocal, gLocal, Ne, padding, numImages, 
        xPos, yPos, zPos, tmpResults);

    for(int i = 0; i < numImages; i++) {
      tmpSendResults[(numImages*j) + i] = tmpResults[i];          
    }//end for i
  }//end for j

  for(int i = 0; i < npesActive; i++) {
    sendSizes[i] = (numImages*(sendSizes[i]/3));
    recvSizes[i] = (numImages*(recvSizes[i]/3));
    sendOff[i] = (numImages*(sendOff[i]/3));
    recvOff[i] = (numImages*(recvOff[i]/3));
  }//end for i

  double* tmpRecvResults = new double[numImages*numPts];
  assert(tmpRecvResults);

  par::Mpi_Alltoallv_sparse<double>(tmpSendResults, recvSizes, recvOff,
      tmpRecvResults, sendSizes, sendOff, commActive);

  for(int i = 0; i < npesActive; i++) {
    tmpSendSizes[i] = 0;
  }//end for i

  for(int i = 0; i < numPts; i++) {
    unsigned int ptIdx = sendOff[part[i]] + tmpSendSizes[part[i]]; 
    for(int j = 0; j < numImages; j++) {
      results[(numImages*i) + j] = tmpRecvResults[ptIdx + j];
    }//end for j
    tmpSendSizes[part[i]] += numImages;
  }//end for i

  delete [] tmpRecvResults;
  tmpRecvResults = NULL;

  delete [] tmpSendResults;
  tmpSendResults = NULL;

  delete [] tmpSendSizes;
  tmpSendSizes = NULL;

  delete [] part;
  part = NULL;

  delete [] sendSizes;
  sendSizes = NULL;

  delete [] recvSizes;
  recvSizes = NULL;

  delete [] sendOff;
  sendOff = NULL;

  delete [] recvOff;
  recvOff = NULL;

  delete [] sendPts;
  sendPts = NULL;

  delete [] recvPts;
  recvPts = NULL;

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
      results[i][j] = evalCubicFnOld(fArr, gArr, Ne, xPos, yPos, zPos);
    }
  }
}


void evalCubicGradFn( const std::vector<ot::TreeNode> & blocks,
    const std::vector<std::vector<double> > & fLocal,
    const std::vector<std::vector<double> > & gLocal,
    int Ne, int padding, int numImages,
    double xPos, double yPos, double zPos, std::vector<double>& results) {

  assert(!(blocks.empty()));

  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    results.assign((numImages*3), 0.0);
  } else {   

    int numBlocks = blocks.size();

    unsigned int maxDepth = blocks[0].getMaxDepth();
    unsigned int regLev = blocks[0].getLevel();
    double h = 1.0/(static_cast<double>(Ne));

    double hFac = (1u << (maxDepth - 1));
    double hFacInv = 1.0/hFac;

    unsigned int xint = static_cast<unsigned int>(xPos*hFac);
    unsigned int yint = static_cast<unsigned int>(yPos*hFac);
    unsigned int zint = static_cast<unsigned int>(zPos*hFac);

    ot::TreeNode searchKey(xint, yint, zint, maxDepth, 3, maxDepth);

    unsigned int searchIdx;
    bool found = seq::maxLowerBound<ot::TreeNode>(blocks, searchKey, searchIdx, NULL, NULL);

    int blockId, xId, yId, zId, nx, ny;

    double xs, ys, zs, xe, ye, ze;

    bool foundBlock = false;

    if( found && ( (blocks[searchIdx].isAncestor(searchKey)) || (blocks[searchIdx] == searchKey) ) ) {
      unsigned int minX = blocks[searchIdx].getX();
      unsigned int minY = blocks[searchIdx].getY();
      unsigned int minZ = blocks[searchIdx].getZ();

      unsigned int maxX = blocks[searchIdx].maxX();
      unsigned int maxY = blocks[searchIdx].maxY();
      unsigned int maxZ = blocks[searchIdx].maxZ();

      double xsPt = hFacInv*(static_cast<double>(minX));
      double ysPt = hFacInv*(static_cast<double>(minY));
      double zsPt = hFacInv*(static_cast<double>(minZ));

      double xePt = hFacInv*(static_cast<double>(maxX));
      double yePt = hFacInv*(static_cast<double>(maxY));
      double zePt = hFacInv*(static_cast<double>(maxZ));

      xs = (xsPt - ((static_cast<double>(padding))*h));
      ys = (ysPt - ((static_cast<double>(padding))*h));
      zs = (zsPt - ((static_cast<double>(padding))*h));

      xe = (xePt + ((static_cast<double>(padding))*h));
      ye = (yePt + ((static_cast<double>(padding))*h));
      ze = (zePt + ((static_cast<double>(padding))*h));

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

      assert( (xPos >= xs) && (xPos <= xe) &&
          (yPos >= ys) && (yPos <= ye) &&
          (zPos >= zs) && (zPos <= ze) );

      blockId = searchIdx;
      nx = (1 + (static_cast<int>((xe - xs)/h)));
      ny = (1 + (static_cast<int>((ye - ys)/h)));
      xId = static_cast<int>((xPos - xs)/h);
      yId = static_cast<int>((yPos - ys)/h);
      zId = static_cast<int>((zPos - zs)/h);

      foundBlock = true;
    } else {
      for(int i = 0; i < numBlocks; i++) {
        unsigned int minX = blocks[i].getX();
        unsigned int minY = blocks[i].getY();
        unsigned int minZ = blocks[i].getZ();

        unsigned int maxX = blocks[i].maxX();
        unsigned int maxY = blocks[i].maxY();
        unsigned int maxZ = blocks[i].maxZ();

        double xsPt = hFacInv*(static_cast<double>(minX));
        double ysPt = hFacInv*(static_cast<double>(minY));
        double zsPt = hFacInv*(static_cast<double>(minZ));

        double xePt = hFacInv*(static_cast<double>(maxX));
        double yePt = hFacInv*(static_cast<double>(maxY));
        double zePt = hFacInv*(static_cast<double>(maxZ));

        xs = (xsPt - ((static_cast<double>(padding))*h));
        ys = (ysPt - ((static_cast<double>(padding))*h));
        zs = (zsPt - ((static_cast<double>(padding))*h));

        xe = (xePt + ((static_cast<double>(padding))*h));
        ye = (yePt + ((static_cast<double>(padding))*h));
        ze = (zePt + ((static_cast<double>(padding))*h));

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

        if( (xPos >= xs) && (xPos <= xe) &&
            (yPos >= ys) && (yPos <= ye) &&
            (zPos >= zs) && (zPos <= ze) ) {
          blockId = i;
          nx = (1 + (static_cast<int>((xe - xs)/h)));
          ny = (1 + (static_cast<int>((ye - ys)/h)));
          xId = static_cast<int>((xPos - xs)/h);
          yId = static_cast<int>((yPos - ys)/h);
          zId = static_cast<int>((zPos - zs)/h);
          foundBlock = true;
          break;
        }
      }
    }

    assert(foundBlock);

    double x0 = xs + ((static_cast<double>(xId))*h);
    double y0 = ys + ((static_cast<double>(yId))*h);
    double z0 = zs + ((static_cast<double>(zId))*h);

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

    results.assign((numImages*3), 0.0);

    if(xPos == xe) {
      xId--;
    }
    if(yPos == ye) {
      yId--;
    }
    if(zPos == ze) {
      zId--;
    }

    for(int j = 0; j < 8; j++) {
      int xx = (xId + (j%2));
      int yy = (yId + ((j/2)%2));
      int zz = (zId + (j/4));
      int pId = ((((zz*ny) + yy)*nx) + xx);
      for(int i = 0; i < numImages; i++) {
        for(int d = 0; d < 3; d++) {
          results[(3*i) + d] += (fLocal[blockId][(numImages*pId) + i]*gradPhiVals[j][0][d]);
          for(int k = 0; k < 3; k++) {
            results[(3*i) + d] += (0.5*h*gLocal[blockId][(((numImages*pId) + i)*3) + k]*gradPhiVals[j][1 + k][d]);
          }
        }
      }
    }

  }

  for(int i = 0; i < (3*numImages); i++) {
    results[i] = results[i]*2.0*static_cast<double>(Ne);
  }

}

double evalCubicFnOld(const double* fArr, const double* gArr, 
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


void evalCubicFn( const std::vector<ot::TreeNode> & blocks,
    const std::vector<std::vector<double> > & fLocal, 
    const std::vector<std::vector<double> > & gLocal,
    int Ne, int padding, int numImages,
    double xPos, double yPos, double zPos, std::vector<double>& results) {

  assert(!(blocks.empty()));

  if( (xPos < 0.0) || (yPos < 0.0) || (zPos < 0.0) ||
      (xPos >= 1.0) || (yPos >= 1.0) || (zPos >= 1.0) )  {
    results.assign(numImages, 0.0); 
  } else {   

    int numBlocks = blocks.size();

    unsigned int maxDepth = blocks[0].getMaxDepth();
    double h = 1.0/(static_cast<double>(Ne));

    double hFac = (1u << (maxDepth - 1));
    double hFacInv = 1.0/hFac;

    unsigned int xint = static_cast<unsigned int>(xPos*hFac);
    unsigned int yint = static_cast<unsigned int>(yPos*hFac);
    unsigned int zint = static_cast<unsigned int>(zPos*hFac);

    ot::TreeNode searchKey(xint, yint, zint, maxDepth, 3, maxDepth);

    unsigned int searchIdx;
    bool found = seq::maxLowerBound<ot::TreeNode>(blocks, searchKey, searchIdx, NULL, NULL);

    int blockId, xId, yId, zId, nx, ny;

    double xs, ys, zs, xe, ye, ze;

    bool foundBlock = false;

    if( found && ( (blocks[searchIdx].isAncestor(searchKey)) || (blocks[searchIdx] == searchKey) ) ) {
      unsigned int minX = blocks[searchIdx].getX();
      unsigned int minY = blocks[searchIdx].getY();
      unsigned int minZ = blocks[searchIdx].getZ();

      unsigned int maxX = blocks[searchIdx].maxX();
      unsigned int maxY = blocks[searchIdx].maxY();
      unsigned int maxZ = blocks[searchIdx].maxZ();

      double xsPt = hFacInv*(static_cast<double>(minX));
      double ysPt = hFacInv*(static_cast<double>(minY));
      double zsPt = hFacInv*(static_cast<double>(minZ));

      double xePt = hFacInv*(static_cast<double>(maxX));
      double yePt = hFacInv*(static_cast<double>(maxY));
      double zePt = hFacInv*(static_cast<double>(maxZ));

      xs = (xsPt - ((static_cast<double>(padding))*h));
      ys = (ysPt - ((static_cast<double>(padding))*h));
      zs = (zsPt - ((static_cast<double>(padding))*h));

      xe = (xePt + ((static_cast<double>(padding))*h));
      ye = (yePt + ((static_cast<double>(padding))*h));
      ze = (zePt + ((static_cast<double>(padding))*h));

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

      assert( (xPos >= xs) && (xPos <= xe) &&
          (yPos >= ys) && (yPos <= ye) &&
          (zPos >= zs) && (zPos <= ze) );

      blockId = searchIdx;
      nx = (1 + (static_cast<int>((xe - xs)/h)));
      ny = (1 + (static_cast<int>((ye - ys)/h)));
      xId = static_cast<int>((xPos - xs)/h);
      yId = static_cast<int>((yPos - ys)/h);
      zId = static_cast<int>((zPos - zs)/h);

      foundBlock = true;
    } else {
      for(int i = 0; i < numBlocks; i++) {
        unsigned int minX = blocks[i].getX();
        unsigned int minY = blocks[i].getY();
        unsigned int minZ = blocks[i].getZ();

        unsigned int maxX = blocks[i].maxX();
        unsigned int maxY = blocks[i].maxY();
        unsigned int maxZ = blocks[i].maxZ();

        double xsPt = hFacInv*(static_cast<double>(minX));
        double ysPt = hFacInv*(static_cast<double>(minY));
        double zsPt = hFacInv*(static_cast<double>(minZ));

        double xePt = hFacInv*(static_cast<double>(maxX));
        double yePt = hFacInv*(static_cast<double>(maxY));
        double zePt = hFacInv*(static_cast<double>(maxZ));

        xs = (xsPt - ((static_cast<double>(padding))*h));
        ys = (ysPt - ((static_cast<double>(padding))*h));
        zs = (zsPt - ((static_cast<double>(padding))*h));

        xe = (xePt + ((static_cast<double>(padding))*h));
        ye = (yePt + ((static_cast<double>(padding))*h));
        ze = (zePt + ((static_cast<double>(padding))*h));

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

        if( (xPos >= xs) && (xPos <= xe) &&
            (yPos >= ys) && (yPos <= ye) &&
            (zPos >= zs) && (zPos <= ze) ) {
          blockId = i;
          nx = (1 + (static_cast<int>((xe - xs)/h)));
          ny = (1 + (static_cast<int>((ye - ys)/h)));
          xId = static_cast<int>((xPos - xs)/h);
          yId = static_cast<int>((yPos - ys)/h);
          zId = static_cast<int>((zPos - zs)/h);
          foundBlock = true;
          break;
        }
      }
    }

    assert(foundBlock);

    double x0 = xs + ((static_cast<double>(xId))*h);
    double y0 = ys + ((static_cast<double>(yId))*h);
    double z0 = zs + ((static_cast<double>(zId))*h);

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

    results.assign(numImages, 0.0);

    if(xPos == xe) {
      xId--;
    }
    if(yPos == ye) {
      yId--;
    }
    if(zPos == ze) {
      zId--;
    }

    for(int j = 0; j < 8; j++) {
      int xx = (xId + (j%2));
      int yy = (yId + ((j/2)%2));
      int zz = (zId + (j/4));
      int pId = ((((zz*ny) + yy)*nx) + xx);
      for(int i = 0; i < numImages; i++) {
        results[i] += (fLocal[blockId][(numImages*pId) + i]*phiVals[j][0]);
        for(int k = 0; k < 3; k++) {
          results[i] += (0.5*h*gLocal[blockId][(((numImages*pId) + i)*3) + k]*phiVals[j][1 + k]);
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



