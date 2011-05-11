
#include "mpi.h"
#include "sys.h"
#include "omg.h"
#include "registration.h"
#include "regInterpCubic.h"
#include "externVars.h"
#include "dendro.h"

#define __PI__ 3.14159265

//Don't want time to be synchronized. Need to check load imbalance.
#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
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

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, "options", 0);

  ot::RegisterEvents();

  PetscLogEventRegister(&optEvent, "Gauss-Newton", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&copyValsToPatchesEvent, "copyValsToPatches", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&meshPatchesEvent, "meshPatches", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&expandPatchesEvent, "expandPatches", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&createPatchesEvent, "createPatches", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&evalObjEvent, "Obj", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&evalGradEvent, "GradObj", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&computeSigEvent, "Sig", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&computeTauEvent, "Tau", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&computeGradTauEvent, "GradTau", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&computeNodalTauEvent, "NodalTau", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&computeNodalGradTauEvent, "NodalGradTau", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&elasMultEvent, "ElasMatMult", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&hessMultEvent, "HessMatMult", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&hessFinestMultEvent, "HessMatMultFinest", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&updateHessContextEvent, "UpdateHess", PETSC_VIEWER_COOKIE);
  PetscLogEventRegister(&createHessContextEvent, "CreateHess", PETSC_VIEWER_COOKIE);

  MPI_Comm commAll = MPI_COMM_WORLD;

  int rank, npesAll;
  MPI_Comm_rank(commAll, &rank);
  MPI_Comm_size(commAll, &npesAll);

  if(argc < 3) {
    if(!rank) {
      std::cout<<"Usage: exe fixedImg movingImg"<<std::endl;
    }
    PetscFinalize();
    exit(0);
  }

  unsigned int dim = 3;
  unsigned int dof = 3;  
  bool incCorner = true;  

  PetscTruth compressLut;
  PetscOptionsHasName(0, "-compressLut", &compressLut);

  if(compressLut) {
    if(!rank) {
      std::cout<<"Mesh is Compressed."<<std::endl;
    }
  }

  PetscReal imgPatchWidth;
  PetscTruth patchWidthSetFromOption;
  PetscOptionsHasName(0, "-imgPatchWidth", &patchWidthSetFromOption);
  PetscOptionsGetReal(0, "-imgPatchWidth", &imgPatchWidth, 0);

  PetscInt maxDepth = 30;
  PetscOptionsGetInt(0, "-maxDepth", &maxDepth, 0);

  PetscInt patchWidthFac = 4;
  PetscOptionsGetInt(0, "-patchWidthFac", &patchWidthFac, 0);

  PetscReal mu = 1.0;
  PetscOptionsGetReal(0, "-mu", &mu, 0);

  PetscReal lambda = 4.0;
  PetscOptionsGetReal(0, "-lambda", &lambda, 0);

  PetscReal alpha = 1.0;
  PetscOptionsGetReal(0, "-alpha", &alpha, 0);

  PetscReal mgLoadFac = 2.0;
  PetscOptionsGetReal(0, "-mgLoadFac", &mgLoadFac, 0);

  PetscReal threshold = 1.0;
  PetscOptionsGetReal(0, "-threshold", &threshold, 0);

  ot::DAMG_Initialize(commAll);

  //Functions for using KSP_Shell (will be used @ the coarsest grid if not all
  //processors are active on the coarsest grid)
  ot::getPrivateMatricesForKSP_Shell = getPrivateMatricesForKSP_Shell_Hess;

  //Set function pointers so that PC_BlockDiag could be used.
  ot::getDofAndNodeSizeForPC_BlockDiag = getDofAndNodeSizeForHessMat;
  ot::computeInvBlockDiagEntriesForPC_BlockDiag = computeInvBlockDiagEntriesForHessMat;

  double**** LaplacianStencil; 
  double**** GradDivStencil;
  double****** PhiMatStencil; 

  PetscInt numGpts = 4;
  PetscOptionsGetInt(0, "-numGpts", &numGpts, 0);

  double* gWts = new double[numGpts];
  double* gPts = new double[numGpts];

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

  createLmat(LaplacianStencil);
  createGDmat(GradDivStencil);
  createPhimat(PhiMatStencil, numGpts, gPts);

  int Nfe;
  std::vector<double> sigImgFinest;
  std::vector<double> tauImgFinest;

  if(!rank) {
    struct dsr hdrSig;
    struct dsr hdrTau;

    readImage(argv[1], &hdrSig, sigImgFinest);
    readImage(argv[2], &hdrTau, tauImgFinest);

    Nfe = hdrSig.dime.dim[1];

    assert(hdrSig.dime.dim[2] == Nfe);
    assert(hdrSig.dime.dim[3] == Nfe);
    assert(hdrTau.dime.dim[1] == Nfe);
    assert(hdrTau.dime.dim[2] == Nfe);
    assert(hdrTau.dime.dim[3] == Nfe);
  }
  par::Mpi_Bcast<int>(&Nfe, 1, 0, commAll);

  PetscTruth useMultiscale;
  PetscOptionsHasName(0, "-useMultiscale", &useMultiscale);

  int Nce = 16;
  if(useMultiscale) {
    if(!rank) {
      std::cout<<"Using Multiscale Continuation..."<<std::endl;
    }
  } else {
    Nce = Nfe;
  }

  PetscInt maxIterCnt = 10;
  PetscOptionsGetInt(0, "-gnMaxIterCnt", &maxIterCnt, 0);

  PetscReal fTol = 0.001;
  PetscOptionsGetReal(0, "-gnFntol", &fTol, 0);
  fTol = fTol/static_cast<double>(Nce);

  PetscReal xTol = 0.1;
  PetscOptionsGetReal(0, "-gnXtol", &xTol, 0);
  xTol = xTol/static_cast<double>(Nce);

  assert(Nfe >= Nce);
  int numOptLevels = (binOp::fastLog2(Nfe/Nce)) + 1;

  MPI_Comm* commCurr = new MPI_Comm[numOptLevels];
  int* npesCurr = new int[numOptLevels];

  //Loop coarsest to finest
  for(int lev = 0; lev < numOptLevels; lev++) {
    int Ne = Nce*(1u << lev);

    npesCurr[lev] = npesAll;
    while(!(foundValidDApart(Ne + 1, npesCurr[lev]))) {
      npesCurr[lev]--;
      if(npesCurr[lev] == 0) {
        break;
      }
    }

    if(!rank) {
      std::cout<<"Multiscale opt lev "<<lev<<" uses "
        <<npesCurr[lev]<<" processors."<<std::endl;
    }

    assert(npesCurr[lev]);
    if(npesCurr[lev] < npesAll) {
      par::splitCommUsingSplittingRank(npesCurr[lev], (commCurr + lev), commAll);
    } else {
      commCurr[lev] = commAll;
    }
  }//end for lev

  assert(npesCurr[numOptLevels - 1] == npesAll);

  std::vector<double>* sigElemental = new std::vector<double>[numOptLevels];
  std::vector<double>* tauElemental = new std::vector<double>[numOptLevels];

  std::vector<std::vector<double> >* sigGlobal = new std::vector<std::vector<double> >[numOptLevels];
  std::vector<std::vector<double> >* gradSigGlobal = new std::vector<std::vector<double> >[numOptLevels];
  std::vector<std::vector<double> >* tauGlobal = new std::vector<std::vector<double> >[numOptLevels];
  std::vector<std::vector<double> >* gradTauGlobal = new std::vector<std::vector<double> >[numOptLevels];

  DA* da1dof = new DA[numOptLevels];

  std::vector<double> sigImgPrev;
  std::vector<double> tauImgPrev;

  //Loop finest to coarsest
  for(int lev = (numOptLevels - 1); lev >= 0; lev--) {
    int Ne = Nce*(1u << lev);

    std::vector<double> sigImgCurr;
    std::vector<double> tauImgCurr;

    if(!rank) {
      if(Ne == Nfe) {
        sigImgCurr = sigImgFinest;
        sigImgFinest.clear();

        tauImgCurr = tauImgFinest;
        tauImgFinest.clear();
      } else {
        coarsenImage((2*Ne), sigImgPrev, sigImgCurr);
        coarsenImage((2*Ne), tauImgPrev, tauImgCurr);
      }
    }

    sigImgPrev = sigImgCurr;
    tauImgPrev = tauImgCurr;

    if(rank < npesCurr[lev]) {

      DACreate3d(commCurr[lev], DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
          PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
          1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, da1dof + lev);

      DA da3dof;
      DACreate3d(commCurr[lev], DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
          PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
          3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da3dof);

      Vec sigN0;
      Vec tauN0;
      createSeqNodalImageVec(Ne, rank, npesCurr[lev], sigImgCurr, sigN0, commCurr[lev]);
      createSeqNodalImageVec(Ne, rank, npesCurr[lev], tauImgCurr, tauN0, commCurr[lev]);

      sigImgCurr.clear();
      tauImgCurr.clear();

      Vec sigNatural, tauNatural;
      createImgN0ToNatural(da1dof[lev], sigN0, tauN0, sigNatural, tauNatural, commCurr[lev]);
      VecDestroy(sigN0);
      VecDestroy(tauN0);

      newProcessImgNatural(da1dof[lev], da3dof, Ne, sigNatural, tauNatural,
          sigGlobal[lev], gradSigGlobal[lev], tauGlobal[lev],
          gradTauGlobal[lev], sigElemental[lev], tauElemental[lev]);

      DADestroy(da3dof);

      VecDestroy(sigNatural);
      VecDestroy(tauNatural);

    }//end if active

  }//end for lev

  sigImgPrev.clear();
  tauImgPrev.clear();

  ot::DAMG *damgPrev = NULL;    
  std::vector<double> dispOct;

  //Loop coarsest to finest
  for(int lev = 0; lev < numOptLevels; lev++) {
    int Ne = Nce*(1u << lev);

    if(!rank) {
      std::cout<<"Starting Lev: "<<lev<<std::endl;
    }

    ot::DAMG *damg = NULL;    

    if(rank < npesCurr[lev]) {
      PetscInt xs, ys, zs, nx, ny, nz;
      DAGetCorners(da1dof[lev], &xs, &ys, &zs, &nx, &ny, &nz);

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

      //Image to octree
      std::vector<ot::TreeNode> linSigOct;
      std::vector<ot::TreeNode> linTauOct;
      std::vector<ot::TreeNode> linOct;
      ot::regularGrid2Octree(sigElemental[lev], Ne, nxe, nye, nze, xs, ys, zs,
          linSigOct, dim, maxDepth, threshold, commCurr[lev]);
      ot::regularGrid2Octree(tauElemental[lev], Ne, nxe, nye, nze, xs, ys, zs,
          linTauOct, dim, maxDepth, threshold, commCurr[lev]);

      sigElemental[lev].clear();
      tauElemental[lev].clear();

      unsigned int locSigSz = linSigOct.size();
      unsigned int locTauSz = linTauOct.size();
      unsigned int globSigSz;
      unsigned int globTauSz;
      par::Mpi_Allreduce<unsigned int>(&locSigSz, &globSigSz, 1, MPI_SUM, commCurr[lev]);
      par::Mpi_Allreduce<unsigned int>(&locTauSz, &globTauSz, 1, MPI_SUM, commCurr[lev]);

      if(!rank) {
        std::cout<<"Lev: "<<lev<<" globSigSz: "<<globSigSz<<" globTauSz: "<<globTauSz<<std::endl;
      }

      if(globTauSz > globSigSz) {
        MPI_Comm tmpComm;
        par::splitComm2way(linTauOct.empty(), &tmpComm, commCurr[lev]);
        if(!(linTauOct.empty())) {
          int tmpNpes, tmpRank;
          MPI_Comm_size(tmpComm, &tmpNpes);
          MPI_Comm_rank(tmpComm, &tmpRank);
          unsigned int avgSz = globSigSz/tmpNpes;
          unsigned int extra = globSigSz%tmpNpes;
          std::vector<ot::TreeNode> tmpLinOct;
          if(tmpRank < extra) {
            par::scatterValues<ot::TreeNode>(linSigOct, tmpLinOct, (avgSz + 1), commCurr[lev]);
          } else {
            par::scatterValues<ot::TreeNode>(linSigOct, tmpLinOct, avgSz, commCurr[lev]);
          }
          ot::mergeOctrees(linTauOct, tmpLinOct, linOct, tmpComm);
        } else {
          std::vector<ot::TreeNode> tmpLinOct;
          par::scatterValues<ot::TreeNode>(linSigOct, tmpLinOct, 0, commCurr[lev]);
        }
      } else {
        MPI_Comm tmpComm;
        par::splitComm2way(linSigOct.empty(), &tmpComm, commCurr[lev]);
        if(!(linSigOct.empty())) {
          int tmpNpes, tmpRank;
          MPI_Comm_size(tmpComm, &tmpNpes);
          MPI_Comm_rank(tmpComm, &tmpRank);
          unsigned int avgSz = globTauSz/tmpNpes;
          unsigned int extra = globTauSz%tmpNpes;
          std::vector<ot::TreeNode> tmpLinOct;
          if(tmpRank < extra) {
            par::scatterValues<ot::TreeNode>(linTauOct, tmpLinOct, (avgSz + 1), commCurr[lev]);
          } else {
            par::scatterValues<ot::TreeNode>(linTauOct, tmpLinOct, avgSz, commCurr[lev]);
          }
          ot::mergeOctrees(linSigOct, tmpLinOct, linOct, tmpComm);
        } else {
          std::vector<ot::TreeNode> tmpLinOct;
          par::scatterValues<ot::TreeNode>(linTauOct, tmpLinOct, 0, commCurr[lev]);
        }
      }
      linSigOct.clear();
      linTauOct.clear();

      std::vector<ot::TreeNode> balOct;
      ot::balanceOctree(linOct, balOct, dim, maxDepth, incCorner, commCurr[lev]);
      linOct.clear();

      int nlevels = 1; 
      PetscInt nlevelsPetscInt = nlevels;
      PetscOptionsGetInt(0, "-nlevels", &nlevelsPetscInt, 0);
      nlevels = nlevelsPetscInt;

      ot::DAMGCreateAndSetDA(commCurr[lev], nlevels, NULL, &damg, 
          balOct, dof, mgLoadFac, compressLut, incCorner);
      balOct.clear();

      ot::PrintDAMG(damg);

      ot::DAMGCreateSuppressedDOFs(damg);

      std::vector<ot::TreeNode> imgPatches;
      std::vector<unsigned int> mesh;

      createImagePatches(Ne, DAMGGetDA(damg), imgPatches);

      if(!patchWidthSetFromOption) {
        imgPatchWidth = patchWidthFac/static_cast<double>(Ne);
      }

      expandImagePatches(Ne, imgPatchWidth, ((DAMGGetDA(damg))->getBlocks()), imgPatches);

      meshImagePatches(imgPatches, mesh);

      std::vector<std::vector<double> > sigLocal;
      std::vector<std::vector<double> > gradSigLocal;
      std::vector<std::vector<double> > tauLocal;
      std::vector<std::vector<double> > gradTauLocal;

      copyValuesToImagePatches(da1dof[lev], imgPatches, sigGlobal[lev], gradSigGlobal[lev],
          tauGlobal[lev], gradTauGlobal[lev], sigLocal,
          gradSigLocal, tauLocal, gradTauLocal);

      DADestroy(da1dof[lev]);

      sigGlobal[lev].clear();
      gradSigGlobal[lev].clear();
      tauGlobal[lev].clear();
      gradTauGlobal[lev].clear();

      createNewHessContexts(damg, imgPatchWidth, imgPatches, mesh,
          sigLocal, gradSigLocal, tauLocal, gradTauLocal,
          PhiMatStencil, LaplacianStencil, GradDivStencil,
          numGpts, gWts, gPts, mu, lambda, alpha);

      sigLocal.clear();
      gradSigLocal.clear();
      tauLocal.clear();
      gradTauLocal.clear();

      imgPatches.clear();
      mesh.clear();
    }//end if active

    ot::DA* dao = NULL;
    std::vector<double> nodePts;
    if(rank < npesCurr[lev]) {
      dao = DAMGGetDA(damg);
      if(lev) {
        double hFac = 1.0/static_cast<double>(1u << maxDepth);
        if(dao->iAmActive()) {
          for(dao->init<ot::DA_FLAGS::WRITABLE>();
              dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
              dao->next<ot::DA_FLAGS::WRITABLE>()) {
            Point pt = dao->getCurrentOffset();
            unsigned int xint = pt.xint();
            unsigned int yint = pt.yint();
            unsigned int zint = pt.zint();
            double x0 = hFac*static_cast<double>(xint);
            double y0 = hFac*static_cast<double>(yint);
            double z0 = hFac*static_cast<double>(zint);
            unsigned int idx = dao->curr();
            unsigned char hnMask = dao->getHangingNodeIndex(idx);
            if(!(hnMask & 1)) {
              nodePts.push_back(x0);
              nodePts.push_back(y0);
              nodePts.push_back(z0);
            }//end if hanging anchor
          }//end WRITABLE
        }//end if active
      }//end if coarsest
    }//end if active

    std::vector<double> nodeVals;
    if(lev) {
      int numLocalPts = nodePts.size()/3;
      int numGlobalPts;
      par::Mpi_Allreduce<int>(&numLocalPts, &numGlobalPts, 1, MPI_SUM, commAll);

      int newAvgSize = numGlobalPts/(npesCurr[lev - 1]);
      int newExtra = numGlobalPts%(npesCurr[lev - 1]);

      std::vector<double> nodePtsDup;
      if( rank >= npesCurr[lev - 1]) {
        par::scatterValues<double>(nodePts, nodePtsDup, 0, commAll);
      } else if (rank < newExtra) {
        par::scatterValues<double>(nodePts, nodePtsDup, (3*(newAvgSize + 1)), commAll);
      } else {
        par::scatterValues<double>(nodePts, nodePtsDup, (3*newAvgSize), commAll);
      }
      nodePts.clear();

      std::vector<double> nodeValsDup;
      if(rank < npesCurr[lev - 1]) {
        ot::interpolateData(DAMGGetDA(damgPrev), dispOct, nodeValsDup, NULL, 3, nodePtsDup);
      }
      nodePtsDup.clear();

      par::scatterValues<double>(nodeValsDup, nodeVals, (3*numLocalPts), commAll);
      nodeValsDup.clear();
    }//end if coarsest

    dispOct.clear();
    if(damgPrev) {
      ot::DAMGDestroy(damgPrev);
      damgPrev = NULL;
    }
    damgPrev = damg;

    if(rank < npesCurr[lev]) {
      Vec Uin;
      dao->createVector(Uin, false, false, 3);
      VecZeroEntries(Uin);

      if(lev) {
        PetscScalar* inArr;
        dao->vecGetBuffer(Uin, inArr, false, false, false, 3);

        int ptsCtr = 0;
        if(dao->iAmActive()) {
          for(dao->init<ot::DA_FLAGS::WRITABLE>();
              dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
              dao->next<ot::DA_FLAGS::WRITABLE>()) {
            unsigned int idx = dao->curr();
            unsigned char hnMask = dao->getHangingNodeIndex(idx);
            if(!(hnMask & 1)) {
              for(int d = 0; d < 3; d++) {
                inArr[(3*idx) + d] = nodeVals[(3*ptsCtr) + d];
              }
              ptsCtr++;
            }//end if hanging anchor
          }//end WRITABLE
        }//end if active

        dao->vecRestoreBuffer(Uin, inArr, false, false, false, 3);
      }//end if coarsest
      nodeVals.clear();

      Vec Uout;
      VecDuplicate(Uin, &Uout);
      newGaussNewton(damg, fTol, xTol, maxIterCnt, imgPatchWidth, Uin, Uout);

      fTol = 0.5*fTol;
      xTol = 0.5*xTol;
      maxIterCnt = maxIterCnt + 10;

      destroyHessContexts(damg);

      VecDestroy(Uin);

      PetscScalar* octDispVals;
      VecGetArray(Uout, &octDispVals);

      PetscInt dispOctSz;
      VecGetLocalSize(Uout, &dispOctSz);

      dispOct.resize(dispOctSz);
      for(int i = 0; i < dispOctSz; i++) {
        dispOct[i] = octDispVals[i];
      }

      VecRestoreArray(Uout, &octDispVals);

      VecDestroy(Uout);
    }//end if active

  }//end for lev

  delete [] commCurr;
  delete [] npesCurr;

  delete [] sigElemental;
  delete [] tauElemental;

  delete [] sigGlobal;
  delete [] tauGlobal;
  delete [] gradSigGlobal;
  delete [] gradTauGlobal;

  delete [] da1dof;

  destroyLmat(LaplacianStencil);
  destroyGDmat(GradDivStencil);
  destroyPhimat(PhiMatStencil, numGpts);

  delete [] gPts;
  delete [] gWts;

  DA dar;
  DACreate3d(commAll, DA_NONPERIODIC, DA_STENCIL_BOX, Nfe + 1, Nfe + 1, Nfe + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
      3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dar);

  Vec UrgGlobal;
  mortonToRgGlobalDisp(damgPrev, dar, Nfe, dispOct, UrgGlobal);

  dispOct.clear();
  ot::DAMGDestroy(damgPrev);

  ot::DAMG_Finalize();

  PetscTruth saveFinalRgDisp;
  PetscOptionsHasName(0, "-saveFinalRgDisp", &saveFinalRgDisp);

  if(saveFinalRgDisp) {
    Vec UrgNatural;
    DACreateNaturalVector(dar, &UrgNatural);

    DAGlobalToNaturalBegin(dar, UrgGlobal, INSERT_VALUES, UrgNatural);
    DAGlobalToNaturalEnd(dar, UrgGlobal, INSERT_VALUES, UrgNatural);

    char fname[256];
    sprintf(fname, "%s_Disp_%d_%d.dat", argv[1], rank, npesAll);
    saveVector(UrgNatural, fname);

    VecDestroy(UrgNatural);
  }

  double maxDetJac, minDetJac;
  detJacMaxAndMin(dar, UrgGlobal, &maxDetJac, &minDetJac);

  if(!rank) {
    std::cout<<" Max Det Jac: "<<maxDetJac<<" Min Det Jac: "<<minDetJac<<std::endl;
  }

  VecDestroy(UrgGlobal);

  DADestroy(dar);

  PetscFinalize();

}



