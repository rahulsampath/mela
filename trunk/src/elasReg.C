
#include "mpi.h"
#include "sys/sys.h"
#include "omg/omg.h"
#include "registration.h"
#include "regInterpCubic.h"
#include "externVars.h"
#include "dendro.h"
#include "elasReg.h"
#include <cstring>
#include "externRegVars.h"
#include <sys/time.h>

#define __PI__ 3.14159265

//Don't want time to be synchronized. Need to check load imbalance.
#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

extern struct timeval  tp;
extern struct timezone tzp;
#define get_seconds()   (gettimeofday(&tp, &tzp), \
                        (double)tp.tv_sec + (double)tp.tv_usec / 1000000.0)



double elasRunOpt(char *targetImg, char *tmpImg, char *disp, char *obj)

 {
//  PetscInitialize(&argc, &argv, "options", 0);

  ot::RegisterEvents();

  PetscLogEventRegister("tauElemAtU", PETSC_VIEWER_COOKIE, &tauElemAtUEvent);
  PetscLogEventRegister("Gauss-Newton", PETSC_VIEWER_COOKIE, &optEvent);
  PetscLogEventRegister("createPatches", PETSC_VIEWER_COOKIE, &createPatchesEvent);
  PetscLogEventRegister("Obj", PETSC_VIEWER_COOKIE, &evalObjEvent);
  PetscLogEventRegister("GradObj", PETSC_VIEWER_COOKIE, &evalGradEvent);
  PetscLogEventRegister("Sig", PETSC_VIEWER_COOKIE, &computeSigEvent);
  PetscLogEventRegister("Tau", PETSC_VIEWER_COOKIE, &computeTauEvent);
  PetscLogEventRegister("GradTau", PETSC_VIEWER_COOKIE, &computeGradTauEvent);
  PetscLogEventRegister("NodalTau", PETSC_VIEWER_COOKIE, &computeNodalTauEvent);
  PetscLogEventRegister("NodalGradTau", PETSC_VIEWER_COOKIE, &computeNodalGradTauEvent);
  PetscLogEventRegister("ElasMatMult", PETSC_VIEWER_COOKIE, &elasMultEvent);
  PetscLogEventRegister("HessMatMult", PETSC_VIEWER_COOKIE, &hessMultEvent);
  PetscLogEventRegister("HessMatMultFinest", PETSC_VIEWER_COOKIE, &hessFinestMultEvent);
  PetscLogEventRegister("UpdateHess", PETSC_VIEWER_COOKIE, &updateHessContextEvent);
  PetscLogEventRegister("CreateHess", PETSC_VIEWER_COOKIE, &createHessContextEvent);

  MPI_Comm commAll = MPI_COMM_WORLD;

  int rank, npesAll;
  MPI_Comm_rank(commAll, &rank);
  MPI_Comm_size(commAll, &npesAll);

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

  PetscInt maxDepth = 30;
  PetscOptionsGetInt(0, "-maxDepth", &maxDepth, 0);

  PetscInt padding = 2;
  PetscOptionsGetInt(0, "-padding", &padding, 0);

  int numImages = 1;//This is the number of components in the vector image 

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

  double* gPts;
  double* gWts;

  createGaussPtsAndWts(gPts, gWts, numGpts);

  createLmat(LaplacianStencil);
  createGDmat(GradDivStencil);
  createPhimat(PhiMatStencil, numGpts, gPts);

  int Nfe;
  std::vector<double> sigImgFinest;
  std::vector<double> tauImgFinest;

  if(!rank) {
    struct dsr hdrSig;
    struct dsr hdrTau;

    readImages(targetImg, &hdrSig, numImages, sigImgFinest);
    readImages(tmpImg, &hdrTau, numImages, tauImgFinest);

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

  PetscReal fTolInit = 0.001;
  PetscOptionsGetReal(0, "-gnFntol", &fTolInit, 0);

  PetscReal xTolInit = 0.1;
  PetscOptionsGetReal(0, "-gnXtol", &xTolInit, 0);

  assert(Nfe >= Nce);
  int numOptLevels = (binOp::fastLog2(Nfe/Nce)) + 1;

  MPI_Comm* commCurr = new MPI_Comm[numOptLevels];
  assert(commCurr);

  int* npesCurr = new int[numOptLevels];
  assert(npesCurr);

  double objImgValOut;
 
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
  assert(sigElemental);

  std::vector<double>* tauElemental = new std::vector<double>[numOptLevels];
  assert(tauElemental);

  std::vector<double>* sigGlobal = new std::vector<double>[numOptLevels];
  assert(sigGlobal);

  std::vector<double>* gradSigGlobal = new std::vector<double>[numOptLevels];
  assert(gradSigGlobal);

  std::vector<double>* tauGlobal = new std::vector<double>[numOptLevels];
  assert(tauGlobal);

  std::vector<double>* gradTauGlobal = new std::vector<double>[numOptLevels];
  assert(gradTauGlobal);

  DA* da1dof = new DA[numOptLevels];
  assert(da1dof);

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
        coarsenImage((2*Ne), numImages, sigImgPrev, sigImgCurr);
        coarsenImage((2*Ne), numImages, tauImgPrev, tauImgCurr);
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
      createSeqNodalImageVec(Ne, numImages, rank, npesCurr[lev], sigImgCurr, sigN0, commCurr[lev]);
      createSeqNodalImageVec(Ne, numImages, rank, npesCurr[lev], tauImgCurr, tauN0, commCurr[lev]);

      sigImgCurr.clear();
      tauImgCurr.clear();

      Vec sigNatural, tauNatural;
      createImgN0ToNatural(da1dof[lev], sigN0, tauN0, sigNatural, tauNatural, commCurr[lev]);
      VecDestroy(sigN0);
      VecDestroy(tauN0);

      //This function must be modified for the case of vector images, i.e. if numImages > 1
      processImgNatural(da1dof[lev], da3dof, sigNatural, tauNatural,
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

    double fTol = fTolInit/(static_cast<double>(Ne));
    double xTol = xTolInit/(static_cast<double>(Ne));

    if(!rank) {
      std::cout<<"Starting Lev: "<<lev<<std::endl;
    }

    ot::DAMG *damg = NULL;    

    PetscInt xs = 0;
    PetscInt ys = 0;
    PetscInt zs = 0;
    PetscInt nx = 0;
    PetscInt ny = 0;
    PetscInt nz = 0;
    int nxe = 0;
    int nye = 0;
    int nze = 0;

    if(rank < npesCurr[lev]) {
      DAGetCorners(da1dof[lev], &xs, &ys, &zs, &nx, &ny, &nz);

      nxe = nx;
      nye = ny;
      nze = nz;
      if((xs + nx) == (Ne + 1)) {
        nxe = nx - 1;
      } 
      if((ys + ny) == (Ne + 1)) {
        nye = ny - 1;
      } 
      if((zs + nz) == (Ne + 1)) {
        nze = nz - 1;
      }
    }//end if active

    PetscLogEventBegin(tauElemAtUEvent, 0, 0, 0, 0);

    std::vector<double> rgNodePts;
    if(lev) {
      if(rank < npesCurr[lev]) {
        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              rgNodePts.push_back(static_cast<double>(xi)/static_cast<double>(Ne));
              rgNodePts.push_back(static_cast<double>(yi)/static_cast<double>(Ne));
              rgNodePts.push_back(static_cast<double>(zi)/static_cast<double>(Ne));
            }//end xi
          }//end yi
        }//end zi
      }//end if active
    }//end if coarsest

    std::vector<double> rgNodeVals;
    if(lev) {
      int numLocalPts = (rgNodePts.size())/3;
      int numGlobalPts;
      par::Mpi_Allreduce<int>(&numLocalPts, &numGlobalPts, 1, MPI_SUM, commAll);

      int damgPrevNpesActive;

      if(!rank) {
        assert(damgPrev != NULL);
        assert((DAMGGetDA(damgPrev)) != NULL);
        damgPrevNpesActive = (DAMGGetDA(damgPrev))->getNpesActive();
      }

      par::Mpi_Bcast<int>(&damgPrevNpesActive, 1, 0, commAll);

      int newAvgSize = numGlobalPts/damgPrevNpesActive;
      int newExtra = numGlobalPts%damgPrevNpesActive;

      std::vector<double> rgNodePtsDup;
      if( rank >= damgPrevNpesActive) {
        par::scatterValues<double>(rgNodePts, rgNodePtsDup, 0, commAll);
      } else if (rank < newExtra) {
        par::scatterValues<double>(rgNodePts, rgNodePtsDup, (3*(newAvgSize + 1)), commAll);
      } else {
        par::scatterValues<double>(rgNodePts, rgNodePtsDup, (3*newAvgSize), commAll);
      }
      rgNodePts.clear();

      std::vector<double> rgNodeValsDup;
      if(damgPrev) {
        ot::interpolateData(DAMGGetDA(damgPrev), dispOct, rgNodeValsDup, NULL, 3, rgNodePtsDup);
      }
      rgNodePtsDup.clear();

      par::scatterValues<double>(rgNodeValsDup, rgNodeVals, (3*numLocalPts), commAll);
      rgNodeValsDup.clear();

      assert(rgNodeVals.size() == (3*nxe*nye*nze));
    }//end if coarsest

    //Compute Tau At U using rgNodeVals
    if(lev) {
      if(rank < npesCurr[lev]) {
        //Do Not Free lx, ly, lz. They are managed by DA
        const PetscInt* lx = NULL;
        const PetscInt* ly = NULL;
        const PetscInt* lz = NULL;

        PetscInt npx, npy, npz; 

        DAGetOwnershipRanges(da1dof[lev], &lx, &ly, &lz);
        DAGetInfo(da1dof[lev], PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &npx, &npy, &npz,	
            PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

        assert(lx);
        assert(ly);
        assert(lz);

        std::vector<double> scanLx(npx);
        std::vector<double> scanLy(npy);
        std::vector<double> scanLz(npz);

        scanLx[0] = 0;
        scanLy[0] = 0;
        scanLz[0] = 0;
        for(int i = 1; i < npx; i++) {
          scanLx[i] = scanLx[i - 1] + (static_cast<double>(lx[i - 1])/static_cast<double>(Ne));
        }
        for(int i = 1; i < npy; i++) {
          scanLy[i] = scanLy[i - 1] + (static_cast<double>(ly[i - 1])/static_cast<double>(Ne));
        }
        for(int i = 1; i < npz; i++) {
          scanLz[i] = scanLz[i - 1] + (static_cast<double>(lz[i - 1])/static_cast<double>(Ne));
        }

        int* sendCnts = new int[npesCurr[lev]];
        assert(sendCnts);

        int* part = new int[(nxe*nye*nze)];
        assert(part);

        for(int i = 0; i < npesCurr[lev]; i++) {
          sendCnts[i] = 0;
        }//end for i

        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              int idx = ((((zi - zs)*nye) + (yi - ys))*nxe) + (xi - xs);
              double xPt = (static_cast<double>(xi)/static_cast<double>(Ne)) + rgNodeVals[3*idx];
              double yPt = (static_cast<double>(yi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 1];
              double zPt = (static_cast<double>(zi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 2];
              if( (xPt < 0.0) || (yPt < 0.0) || (zPt < 0.0) ||
                  (xPt >= 1.0) || (yPt >= 1.0) || (zPt >= 1.0) ) {
                part[idx] = rank;
              } else {
                unsigned int xRes, yRes, zRes;
                seq::maxLowerBound<double>(scanLx, xPt, xRes, 0, 0);
                seq::maxLowerBound<double>(scanLy, yPt, yRes, 0, 0);
                seq::maxLowerBound<double>(scanLz, zPt, zRes, 0, 0);
                part[idx] = (((zRes*npy) + yRes)*npx) + xRes;
              }
              assert(part[idx] < npesCurr[lev]);
              sendCnts[part[idx]] += 3;
            }//end for xi
          }//end for yi
        }//end for zi

        scanLx.clear();
        scanLy.clear();
        scanLz.clear();  

        int* recvCnts = new int[npesCurr[lev]];
        assert(recvCnts);

        par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, commCurr[lev]);

        int* sendOffsets = new int[npesCurr[lev]];
        assert(sendOffsets);

        int* recvOffsets = new int[npesCurr[lev]];
        assert(recvOffsets);

        sendOffsets[0] = 0;
        recvOffsets[0] = 0;
        for(int i = 1; i < npesCurr[lev]; i++) {
          sendOffsets[i] = sendOffsets[i - 1] + sendCnts[i - 1];
          recvOffsets[i] = recvOffsets[i - 1] + recvCnts[i - 1];
        }//end for i

        int* tmpSendCnts = new int[npesCurr[lev]];  
        assert(tmpSendCnts);
        for(int i = 0; i < npesCurr[lev]; i++) {
          tmpSendCnts[i] = 0;
        }//end for i

        int totalSend = (3*nxe*nye*nze);
        double* xyzSendVals = new double[totalSend];
        assert(xyzSendVals);

        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              int idx = ((((zi - zs)*nye) + (yi - ys))*nxe) + (xi - xs);
              double xPt = (static_cast<double>(xi)/static_cast<double>(Ne)) + rgNodeVals[3*idx];
              double yPt = (static_cast<double>(yi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 1];
              double zPt = (static_cast<double>(zi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 2];

              int sendId = sendOffsets[part[idx]] + tmpSendCnts[part[idx]];

              xyzSendVals[sendId] = xPt;
              xyzSendVals[sendId + 1] = yPt;
              xyzSendVals[sendId + 2] = zPt;

              tmpSendCnts[part[idx]] += 3;
            }//end for xi
          }//end for yi
        }//end for zi

        int totalRecv = recvOffsets[npesCurr[lev] - 1] + recvCnts[npesCurr[lev] - 1];
        double* xyzRecvVals = new double[totalRecv];
        assert(xyzRecvVals);

        par::Mpi_Alltoallv_sparse<double>(xyzSendVals, sendCnts, sendOffsets,
            xyzRecvVals, recvCnts, recvOffsets, commCurr[lev]);

        DA da3dof;
        DACreate3d(commCurr[lev], DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da3dof);

        Vec tauVec;
        Vec gradTauVec;

        DACreateGlobalVector(da1dof[lev], &tauVec);
        DACreateGlobalVector(da3dof, &gradTauVec);

        PetscScalar* tauArr;
        PetscScalar* gradTauArr;

        VecGetArray(tauVec, &tauArr);
        VecGetArray(gradTauVec, &gradTauArr);

        for(int i = 0; i < (nx*ny*nz); i++) {
          tauArr[i] = tauGlobal[lev][i];
          for(int j = 0; j < 3; j++) {
            gradTauArr[(3*i) + j] = gradTauGlobal[lev][(3*i) + j];
          }//end for j
        }//end for i

        VecRestoreArray(tauVec, &tauArr);
        VecRestoreArray(gradTauVec, &gradTauArr);

        Vec tauGhosted;
        Vec gradTauGhosted;

        DACreateLocalVector(da1dof[lev], &tauGhosted);
        DACreateLocalVector(da3dof, &gradTauGhosted);

        DAGlobalToLocalBegin(da1dof[lev], tauVec, INSERT_VALUES, tauGhosted);
        DAGlobalToLocalEnd(da1dof[lev], tauVec, INSERT_VALUES, tauGhosted);

        DAGlobalToLocalBegin(da3dof, gradTauVec, INSERT_VALUES, gradTauGhosted);
        DAGlobalToLocalEnd(da3dof, gradTauVec, INSERT_VALUES, gradTauGhosted);

        PetscScalar*** tauGhostedArr;
        PetscScalar**** gradTauGhostedArr;

        DAVecGetArray(da1dof[lev], tauGhosted, &tauGhostedArr);
        DAVecGetArrayDOF(da3dof, gradTauGhosted, &gradTauGhostedArr);

        double* sendTauVals = new double[totalRecv/3];
        assert(sendTauVals);

        for(unsigned int i = 0; i < totalRecv; i += 3) {
          double xPt = xyzRecvVals[i];
          double yPt = xyzRecvVals[i + 1];
          double zPt = xyzRecvVals[i + 2];

          //interpolate tau 
          double tauAtUval = 0;

          if( (xPt >= 0.0) && (yPt >= 0.0) && (zPt >= 0.0) 
              && (xPt < 1.0) && (yPt < 1.0) && (zPt < 1.0) ) {
            unsigned int ei = static_cast<int>(xPt*static_cast<double>(Ne));
            unsigned int ej = static_cast<int>(yPt*static_cast<double>(Ne));
            unsigned int ek = static_cast<int>(zPt*static_cast<double>(Ne));

            double x0 = static_cast<double>(ei)/static_cast<double>(Ne);
            double y0 = static_cast<double>(ej)/static_cast<double>(Ne);
            double z0 = static_cast<double>(ek)/static_cast<double>(Ne);

            double psi = ((xPt - x0)*2.0*static_cast<double>(Ne)) - 1.0;
            double eta = ((yPt - y0)*2.0*static_cast<double>(Ne)) - 1.0;
            double gamma = ((zPt - z0)*2.0*static_cast<double>(Ne)) - 1.0;
            double phiVals[8][4];

            evalAll3Dcubic(psi, eta, gamma, phiVals);

            for(int j = 0; j < 8; j++) {
              int xid = ei + (j%2);
              int yid = ej + ((j/2)%2);
              int zid = ek + (j/4);
              tauAtUval += ((tauGhostedArr[zid][yid][xid])*(phiVals[j][0]));
              for(int k = 0; k < 3; k++) {
                tauAtUval += ((0.5/static_cast<double>(Ne))*
                    (gradTauGhostedArr[zid][yid][xid][k])*(phiVals[j][1 + k]));
              }//end for k
            }//end for j

          }//end if in domain

          sendTauVals[i/3] = tauAtUval;
        }//end for i

        DAVecRestoreArray(da1dof[lev], tauGhosted, &tauGhostedArr);
        DAVecRestoreArrayDOF(da3dof, gradTauGhosted, &gradTauGhostedArr);

        VecDestroy(tauVec);
        VecDestroy(gradTauVec);
        VecDestroy(tauGhosted);
        VecDestroy(gradTauGhosted);

        DADestroy(da3dof);

        double* recvTauVals = new double[totalSend/3];
        assert(recvTauVals);

        for(int i = 0; i < npesCurr[lev]; i++) {
          sendCnts[i] = sendCnts[i]/3;
          sendOffsets[i] = sendOffsets[i]/3;
          recvCnts[i] = recvCnts[i]/3;
          recvOffsets[i] = recvOffsets[i]/3;
        }//end for i

        par::Mpi_Alltoallv_sparse<double>(sendTauVals, recvCnts, recvOffsets, 
            recvTauVals, sendCnts, sendOffsets, commCurr[lev]);

        for(int i = 0; i < npesCurr[lev]; i++) {
          tmpSendCnts[i] = 0;
        }//end for i

        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              int idx = ((((zi - zs)*nye) + (yi - ys))*nxe) + (xi - xs);
              int sendId = sendOffsets[part[idx]] + tmpSendCnts[part[idx]];
              tauElemental[lev][idx] = recvTauVals[sendId];
              tmpSendCnts[part[idx]]++;
            }//end for xi
          }//end for yi
        }//end for zi

        assert(sendTauVals);
        delete [] sendTauVals;
        sendTauVals = NULL;

        assert(recvTauVals);
        delete [] recvTauVals;
        recvTauVals = NULL;

        assert(xyzSendVals);
        delete [] xyzSendVals;
        xyzSendVals = NULL;

        assert(xyzRecvVals);
        delete [] xyzRecvVals;
        xyzRecvVals = NULL;

        assert(sendOffsets);
        delete [] sendOffsets;
        sendOffsets = NULL;

        assert(recvOffsets);
        delete [] recvOffsets;
        recvOffsets = NULL;

        assert(tmpSendCnts);
        delete [] tmpSendCnts;
        tmpSendCnts = NULL;

        assert(sendCnts);
        delete [] sendCnts;
        sendCnts = NULL;

        assert(recvCnts);
        delete [] recvCnts;
        recvCnts = NULL;

        assert(part);
        delete [] part;
        part = NULL;

      }//end if active
    }//end if coarsest

    rgNodeVals.clear();

    PetscLogEventEnd(tauElemAtUEvent, 0, 0, 0, 0);

    if(rank < npesCurr[lev]) {
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

      bool balOctEmpty = balOct.empty();

      MPI_Comm balOctComm;
      par::splitComm2way(balOctEmpty, &balOctComm, commCurr[lev]);

      if(!balOctEmpty) {
        int balOctNpes;
        MPI_Comm_size(balOctComm, &balOctNpes);
        assert(rank < balOctNpes);
        ot::DAMGCreateAndSetDA(balOctComm, nlevels, NULL, &damg, 
            balOct, dof, mgLoadFac, compressLut, incCorner);
      }
      balOct.clear();

      std::vector<std::vector<double> > sigLocal;
      std::vector<std::vector<double> > gradSigLocal;
      std::vector<std::vector<double> > tauLocal;
      std::vector<std::vector<double> > gradTauLocal;

      ot::DA* thisDA = NULL;

      if(damg) {
        ot::PrintDAMG(damg);
        ot::DAMGCreateSuppressedDOFs(damg);
        thisDA = DAMGGetDA(damg);
      }

      createImagePatches(da1dof[lev], thisDA, padding, numImages,
          sigGlobal[lev], tauGlobal[lev], gradSigGlobal[lev], gradTauGlobal[lev],
          sigLocal, tauLocal, gradSigLocal, gradTauLocal);

      DADestroy(da1dof[lev]);

      sigGlobal[lev].clear();
      gradSigGlobal[lev].clear();
      tauGlobal[lev].clear();
      gradTauGlobal[lev].clear();

      if(damg) {
        createHessContexts(damg, Ne, padding, numImages,
            sigLocal, gradSigLocal, tauLocal, gradTauLocal,
            PhiMatStencil, LaplacianStencil, GradDivStencil, 
            numGpts, gWts, gPts, mu, lambda, alpha);
      }

      sigLocal.clear();
      gradSigLocal.clear();
      tauLocal.clear();
      gradTauLocal.clear();

    }//end if active

    ot::DA* dao = NULL;
    std::vector<double> nodePts;
    if(damg) {
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

      int damgPrevNpesActive;

      if(!rank) {
        assert(damgPrev != NULL);
        assert((DAMGGetDA(damgPrev)) != NULL);
        damgPrevNpesActive = (DAMGGetDA(damgPrev))->getNpesActive();
      }

      par::Mpi_Bcast<int>(&damgPrevNpesActive, 1, 0, commAll);

      int newAvgSize = numGlobalPts/damgPrevNpesActive;
      int newExtra = numGlobalPts%damgPrevNpesActive;

      std::vector<double> nodePtsDup;
      if( rank >= damgPrevNpesActive) {
        par::scatterValues<double>(nodePts, nodePtsDup, 0, commAll);
      } else if (rank < newExtra) {
        par::scatterValues<double>(nodePts, nodePtsDup, (3*(newAvgSize + 1)), commAll);
      } else {
        par::scatterValues<double>(nodePts, nodePtsDup, (3*newAvgSize), commAll);
      }
      nodePts.clear();

      std::vector<double> nodeValsDup;
      if(damgPrev) {
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

    if(damg) {
      Vec Uin;
      dao->createVector(Uin, false, false, 3);
      VecZeroEntries(Uin);

      if(lev) {
        int ptsCtr = 0;
        if(dao->iAmActive()) {
          PetscScalar* inArr;
          dao->vecGetBuffer(Uin, inArr, false, false, false, 3);

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

          dao->vecRestoreBuffer(Uin, inArr, false, false, false, 3);
        }//end if active
      }//end if coarsest
      nodeVals.clear();

      Vec Uout;
      VecDuplicate(Uin, &Uout);

      if(!rank) {
        std::cout<<"Starting GaussNewton..."<<std::endl;
      }

      objImgValOut= gaussNewton(damg, fTol, xTol, maxIterCnt, Uin, Uout);
  
  	if(!rank) {
		  std::cout << "Objective Image Value: " << objImgValOut << std::endl;
    }
    
      destroyHessContexts(damg);

      VecDestroy(Uin);

      PetscScalar* octDispVals;
      VecGetArray(Uout, &octDispVals);

      PetscInt dispOctSz;
      VecGetLocalSize(Uout, &dispOctSz);

      dispOct.resize(dispOctSz);
      for(int i = 0; i < dispOctSz; i++) {
        dispOct[i] = octDispVals[i];
      }//end for i

      VecRestoreArray(Uout, &octDispVals);

      VecDestroy(Uout);
    }//end if active

  }//end for lev

  assert(commCurr);
  delete [] commCurr;
  commCurr = NULL;

  assert(npesCurr);
  delete [] npesCurr;
  npesCurr = NULL;

  assert(sigElemental);
  delete [] sigElemental;
  sigElemental = NULL;

  assert(tauElemental);
  delete [] tauElemental;
  tauElemental = NULL;

  assert(sigGlobal);
  delete [] sigGlobal;
  sigGlobal = NULL;

  assert(tauGlobal);
  delete [] tauGlobal;
  tauGlobal = NULL;

  assert(gradSigGlobal);
  delete [] gradSigGlobal;
  gradSigGlobal = NULL;

  assert(gradTauGlobal);
  delete [] gradTauGlobal;
  gradTauGlobal = NULL;

  assert(da1dof);
  delete [] da1dof;
  da1dof = NULL;

  destroyLmat(LaplacianStencil);
  destroyGDmat(GradDivStencil);
  destroyPhimat(PhiMatStencil, numGpts);

  destroyGaussPtsAndWts(gPts, gWts);

  DA dar;
  Vec UrgGlobal;

  if(damgPrev) {
    DACreate3d(damgPrev[0]->comm, DA_NONPERIODIC, DA_STENCIL_BOX, Nfe + 1, Nfe + 1, Nfe + 1,
        PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
        3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dar);

    mortonToRgGlobalDisp(damgPrev, dar, Nfe, dispOct, UrgGlobal);
  }//end if active

  dispOct.clear();

  PetscTruth saveFinalRgDisp;
  PetscOptionsHasName(0, "-saveFinalRgDisp", &saveFinalRgDisp);

  if(damgPrev) {

    if(saveFinalRgDisp) {
      Vec UrgNatural;
      DACreateNaturalVector(dar, &UrgNatural);

      DAGlobalToNaturalBegin(dar, UrgGlobal, INSERT_VALUES, UrgNatural);
      DAGlobalToNaturalEnd(dar, UrgGlobal, INSERT_VALUES, UrgNatural);

      char fname[256];
      sprintf(fname, "%s_%d_%d.dat", disp, rank, npesAll);
      saveVector(UrgNatural, fname);

      VecDestroy(UrgNatural);
    }

    double maxDetJac, minDetJac;
    detJacMaxAndMin(dar, UrgGlobal, &maxDetJac, &minDetJac);
    if(!rank) {
      std::cout<<" Max Det Jac: "<<maxDetJac<<" Min Det Jac: "<<minDetJac<<std::endl;
		  std::cout << "Final objective Image Value: " << objImgValOut << std::endl;
    }

    VecDestroy(UrgGlobal);

    DADestroy(dar);

    ot::DAMGDestroy(damgPrev);
    damgPrev = NULL;

  }//end if active

  if (!rank){
      FILE *objFunc= fopen(obj,"w");
      fprintf(objFunc,"%1.4e",objImgValOut);
      fclose(objFunc);
  }

  ot::DAMG_Finalize();
  return objImgValOut;

}

int elasVecRunOpt(char *targetImg, char *tmpImg, char *disp, char *obj)
 {

  ot::RegisterEvents();

  PetscLogEventRegister("tauElemAtU", PETSC_VIEWER_COOKIE, &tauElemAtUEvent);
  PetscLogEventRegister("Gauss-Newton", PETSC_VIEWER_COOKIE, &optEvent);
  PetscLogEventRegister("createPatches", PETSC_VIEWER_COOKIE, &createPatchesEvent);
  PetscLogEventRegister("Obj", PETSC_VIEWER_COOKIE, &evalObjEvent);
  PetscLogEventRegister("GradObj", PETSC_VIEWER_COOKIE, &evalGradEvent);
  PetscLogEventRegister("Sig", PETSC_VIEWER_COOKIE, &computeSigEvent);
  PetscLogEventRegister("Tau", PETSC_VIEWER_COOKIE, &computeTauEvent);
  PetscLogEventRegister("GradTau", PETSC_VIEWER_COOKIE, &computeGradTauEvent);
  PetscLogEventRegister("NodalTau", PETSC_VIEWER_COOKIE, &computeNodalTauEvent);
  PetscLogEventRegister("NodalGradTau", PETSC_VIEWER_COOKIE, &computeNodalGradTauEvent);
  PetscLogEventRegister("ElasMatMult", PETSC_VIEWER_COOKIE, &elasMultEvent);
  PetscLogEventRegister("HessMatMult", PETSC_VIEWER_COOKIE, &hessMultEvent);
  PetscLogEventRegister("HessMatMultFinest", PETSC_VIEWER_COOKIE, &hessFinestMultEvent);
  PetscLogEventRegister("UpdateHess", PETSC_VIEWER_COOKIE, &updateHessContextEvent);
  PetscLogEventRegister("CreateHess", PETSC_VIEWER_COOKIE, &createHessContextEvent);

  MPI_Comm commAll = MPI_COMM_WORLD;

  int rank, npesAll;
  MPI_Comm_rank(commAll, &rank);
  MPI_Comm_size(commAll, &npesAll);

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

  PetscInt maxDepth = 30;
  PetscOptionsGetInt(0, "-maxDepth", &maxDepth, 0);

  PetscInt padding = 2;
  PetscOptionsGetInt(0, "-padding", &padding, 0);

  //The number of components in the vector image 
  PetscInt numImages = 1;
  PetscOptionsGetInt(0, "-numImages", &numImages, 0);

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

  double* gPts;
  double* gWts;

  createGaussPtsAndWts(gPts, gWts, numGpts);

  createLmat(LaplacianStencil);
  createGDmat(GradDivStencil);
  createPhimat(PhiMatStencil, numGpts, gPts);

  int Nfe;
  std::vector<double> sigImgFinest;
  std::vector<double> tauImgFinest;

  if(!rank) {
    struct dsr hdrSig;
    struct dsr hdrTau;

    readImages(targetImg, &hdrSig, numImages, sigImgFinest);
    readImages(tmpImg, &hdrTau, numImages, tauImgFinest);

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

  PetscReal fTolInit = 0.001;
  PetscOptionsGetReal(0, "-gnFntol", &fTolInit, 0);

  PetscReal xTolInit = 0.1;
  PetscOptionsGetReal(0, "-gnXtol", &xTolInit, 0);

  assert(Nfe >= Nce);
  int numOptLevels = (binOp::fastLog2(Nfe/Nce)) + 1;

  MPI_Comm* commCurr = new MPI_Comm[numOptLevels];
  assert(commCurr);

  int* npesCurr = new int[numOptLevels];
  assert(npesCurr);

  double objImgValOut;

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
  assert(sigElemental);

  std::vector<double>* tauElemental = new std::vector<double>[numOptLevels];
  assert(tauElemental);

  std::vector<double>* sigGlobal = new std::vector<double>[numOptLevels];
  assert(sigGlobal);

  std::vector<double>* gradSigGlobal = new std::vector<double>[numOptLevels];
  assert(gradSigGlobal);

  std::vector<double>* tauGlobal = new std::vector<double>[numOptLevels];
  assert(tauGlobal);

  std::vector<double>* gradTauGlobal = new std::vector<double>[numOptLevels];
  assert(gradTauGlobal);

  DA* da1dof = new DA[numOptLevels];
  assert(da1dof);

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
        coarsenImage((2*Ne), numImages, sigImgPrev, sigImgCurr);
        coarsenImage((2*Ne), numImages, tauImgPrev, tauImgCurr);
      }
    }

    sigImgPrev = sigImgCurr;
    tauImgPrev = tauImgCurr;

    if(rank < npesCurr[lev]) {

      DACreate3d(commCurr[lev], DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
          PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
          numImages, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, da1dof + lev);

      DA da3dof;
      DACreate3d(commCurr[lev], DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
          PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
          (3*numImages), 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da3dof);

      Vec sigN0;
      Vec tauN0;
      createSeqNodalImageVec(Ne, numImages, rank, npesCurr[lev], sigImgCurr, sigN0, commCurr[lev]);
      createSeqNodalImageVec(Ne, numImages, rank, npesCurr[lev], tauImgCurr, tauN0, commCurr[lev]);

      sigImgCurr.clear();
      tauImgCurr.clear();

      Vec sigNatural, tauNatural;
      createImgN0ToNatural(da1dof[lev], sigN0, tauN0, sigNatural, tauNatural, commCurr[lev]);
      VecDestroy(sigN0);
      VecDestroy(tauN0);

      processImgNatural(da1dof[lev], da3dof, sigNatural, tauNatural,
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

    double fTol = fTolInit/(static_cast<double>(Ne));
    double xTol = xTolInit/(static_cast<double>(Ne));

    if(!rank) {
      std::cout<<"Starting Lev: "<<lev<<std::endl;
    }

    ot::DAMG *damg = NULL;    

    PetscInt xs = 0;
    PetscInt ys = 0;
    PetscInt zs = 0;
    PetscInt nx = 0;
    PetscInt ny = 0;
    PetscInt nz = 0;
    int nxe = 0;
    int nye = 0;
    int nze = 0;

    if(rank < npesCurr[lev]) {
      DAGetCorners(da1dof[lev], &xs, &ys, &zs, &nx, &ny, &nz);

      nxe = nx;
      nye = ny;
      nze = nz;
      if((xs + nx) == (Ne + 1)) {
        nxe = nx - 1;
      } 
      if((ys + ny) == (Ne + 1)) {
        nye = ny - 1;
      } 
      if((zs + nz) == (Ne + 1)) {
        nze = nz - 1;
      }
    }//end if active

    PetscLogEventBegin(tauElemAtUEvent, 0, 0, 0, 0);

    std::vector<double> rgNodePts;
    if(lev) {
      if(rank < npesCurr[lev]) {
        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              rgNodePts.push_back(static_cast<double>(xi)/static_cast<double>(Ne));
              rgNodePts.push_back(static_cast<double>(yi)/static_cast<double>(Ne));
              rgNodePts.push_back(static_cast<double>(zi)/static_cast<double>(Ne));
            }//end xi
          }//end yi
        }//end zi
      }//end if active
    }//end if coarsest

    std::vector<double> rgNodeVals;
    if(lev) {
      int numLocalPts = (rgNodePts.size())/3;
      int numGlobalPts;
      par::Mpi_Allreduce<int>(&numLocalPts, &numGlobalPts, 1, MPI_SUM, commAll);

      int damgPrevNpesActive;

      if(!rank) {
        assert(damgPrev != NULL);
        assert((DAMGGetDA(damgPrev)) != NULL);
        damgPrevNpesActive = (DAMGGetDA(damgPrev))->getNpesActive();
      }

      par::Mpi_Bcast<int>(&damgPrevNpesActive, 1, 0, commAll);

      int newAvgSize = numGlobalPts/damgPrevNpesActive;
      int newExtra = numGlobalPts%damgPrevNpesActive;

      std::vector<double> rgNodePtsDup;
      if( rank >= damgPrevNpesActive) {
        par::scatterValues<double>(rgNodePts, rgNodePtsDup, 0, commAll);
      } else if (rank < newExtra) {
        par::scatterValues<double>(rgNodePts, rgNodePtsDup, (3*(newAvgSize + 1)), commAll);
      } else {
        par::scatterValues<double>(rgNodePts, rgNodePtsDup, (3*newAvgSize), commAll);
      }
      rgNodePts.clear();

      std::vector<double> rgNodeValsDup;
      if(damgPrev) {
        ot::interpolateData(DAMGGetDA(damgPrev), dispOct, rgNodeValsDup, NULL, 3, rgNodePtsDup);
      }
      rgNodePtsDup.clear();

      par::scatterValues<double>(rgNodeValsDup, rgNodeVals, (3*numLocalPts), commAll);
      rgNodeValsDup.clear();

      assert(rgNodeVals.size() == (3*nxe*nye*nze));
    }//end if coarsest

    //Compute Tau At U using rgNodeVals
    if(lev) {
      if(rank < npesCurr[lev]) {
        //Do Not Free lx, ly, lz. They are managed by DA
        const PetscInt* lx = NULL;
        const PetscInt* ly = NULL;
        const PetscInt* lz = NULL;

        PetscInt npx, npy, npz; 

        DAGetOwnershipRanges(da1dof[lev], &lx, &ly, &lz);
        DAGetInfo(da1dof[lev], PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &npx, &npy, &npz,	
            PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

        assert(lx);
        assert(ly);
        assert(lz);

        std::vector<double> scanLx(npx);
        std::vector<double> scanLy(npy);
        std::vector<double> scanLz(npz);

        scanLx[0] = 0;
        scanLy[0] = 0;
        scanLz[0] = 0;
        for(int i = 1; i < npx; i++) {
          scanLx[i] = scanLx[i - 1] + (static_cast<double>(lx[i - 1])/static_cast<double>(Ne));
        }
        for(int i = 1; i < npy; i++) {
          scanLy[i] = scanLy[i - 1] + (static_cast<double>(ly[i - 1])/static_cast<double>(Ne));
        }
        for(int i = 1; i < npz; i++) {
          scanLz[i] = scanLz[i - 1] + (static_cast<double>(lz[i - 1])/static_cast<double>(Ne));
        }

        int* sendCnts = new int[npesCurr[lev]];
        assert(sendCnts);

        int* part = new int[(nxe*nye*nze)];
        assert(part);

        for(int i = 0; i < npesCurr[lev]; i++) {
          sendCnts[i] = 0;
        }//end for i

        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              int idx = ((((zi - zs)*nye) + (yi - ys))*nxe) + (xi - xs);
              double xPt = (static_cast<double>(xi)/static_cast<double>(Ne)) + rgNodeVals[3*idx];
              double yPt = (static_cast<double>(yi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 1];
              double zPt = (static_cast<double>(zi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 2];
              if( (xPt < 0.0) || (yPt < 0.0) || (zPt < 0.0) ||
                  (xPt >= 1.0) || (yPt >= 1.0) || (zPt >= 1.0) ) {
                part[idx] = rank;
              } else {
                unsigned int xRes, yRes, zRes;
                seq::maxLowerBound<double>(scanLx, xPt, xRes, 0, 0);
                seq::maxLowerBound<double>(scanLy, yPt, yRes, 0, 0);
                seq::maxLowerBound<double>(scanLz, zPt, zRes, 0, 0);
                part[idx] = (((zRes*npy) + yRes)*npx) + xRes;
              }
              assert(part[idx] < npesCurr[lev]);
              sendCnts[part[idx]] += 3;
            }//end for xi
          }//end for yi
        }//end for zi

        scanLx.clear();
        scanLy.clear();
        scanLz.clear();  

        int* recvCnts = new int[npesCurr[lev]];
        assert(recvCnts);

        par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, commCurr[lev]);

        int* sendOffsets = new int[npesCurr[lev]];
        assert(sendOffsets);

        int* recvOffsets = new int[npesCurr[lev]];
        assert(recvOffsets);

        sendOffsets[0] = 0;
        recvOffsets[0] = 0;
        for(int i = 1; i < npesCurr[lev]; i++) {
          sendOffsets[i] = sendOffsets[i - 1] + sendCnts[i - 1];
          recvOffsets[i] = recvOffsets[i - 1] + recvCnts[i - 1];
        }//end for i

        int* tmpSendCnts = new int[npesCurr[lev]];  
        assert(tmpSendCnts);
        for(int i = 0; i < npesCurr[lev]; i++) {
          tmpSendCnts[i] = 0;
        }//end for i

        int totalSend = (3*nxe*nye*nze);
        double* xyzSendVals = new double[totalSend];
        assert(xyzSendVals);

        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              int idx = ((((zi - zs)*nye) + (yi - ys))*nxe) + (xi - xs);
              double xPt = (static_cast<double>(xi)/static_cast<double>(Ne)) + rgNodeVals[3*idx];
              double yPt = (static_cast<double>(yi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 1];
              double zPt = (static_cast<double>(zi)/static_cast<double>(Ne)) + rgNodeVals[(3*idx) + 2];

              int sendId = sendOffsets[part[idx]] + tmpSendCnts[part[idx]];

              xyzSendVals[sendId] = xPt;
              xyzSendVals[sendId + 1] = yPt;
              xyzSendVals[sendId + 2] = zPt;

              tmpSendCnts[part[idx]] += 3;
            }//end for xi
          }//end for yi
        }//end for zi

        int totalRecv = recvOffsets[npesCurr[lev] - 1] + recvCnts[npesCurr[lev] - 1];
        double* xyzRecvVals = new double[totalRecv];
        assert(xyzRecvVals);

        par::Mpi_Alltoallv_sparse<double>(xyzSendVals, sendCnts, sendOffsets,
            xyzRecvVals, recvCnts, recvOffsets, commCurr[lev]);

        DA da3dof;
        DACreate3d(commCurr[lev], DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            (3*numImages), 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da3dof);

        Vec tauVec;
        Vec gradTauVec;

        DACreateGlobalVector(da1dof[lev], &tauVec);
        DACreateGlobalVector(da3dof, &gradTauVec);

        PetscScalar* tauArr;
        PetscScalar* gradTauArr;

        VecGetArray(tauVec, &tauArr);
        VecGetArray(gradTauVec, &gradTauArr);

        for(int i = 0; i < (numImages*nx*ny*nz); i++) {
          tauArr[i] = tauGlobal[lev][i];
          for(int j = 0; j < 3; j++) {
            gradTauArr[(3*i) + j] = gradTauGlobal[lev][(3*i) + j];
          }//end for j
        }//end for i

        VecRestoreArray(tauVec, &tauArr);
        VecRestoreArray(gradTauVec, &gradTauArr);

        Vec tauGhosted;
        Vec gradTauGhosted;

        DACreateLocalVector(da1dof[lev], &tauGhosted);
        DACreateLocalVector(da3dof, &gradTauGhosted);

        DAGlobalToLocalBegin(da1dof[lev], tauVec, INSERT_VALUES, tauGhosted);
        DAGlobalToLocalEnd(da1dof[lev], tauVec, INSERT_VALUES, tauGhosted);

        DAGlobalToLocalBegin(da3dof, gradTauVec, INSERT_VALUES, gradTauGhosted);
        DAGlobalToLocalEnd(da3dof, gradTauVec, INSERT_VALUES, gradTauGhosted);

        PetscScalar**** tauGhostedArr;
        PetscScalar**** gradTauGhostedArr;

        DAVecGetArrayDOF(da1dof[lev], tauGhosted, &tauGhostedArr);
        DAVecGetArrayDOF(da3dof, gradTauGhosted, &gradTauGhostedArr);

        double* sendTauVals = new double[numImages*(totalRecv/3)];
        assert(sendTauVals);

        for(unsigned int i = 0; i < totalRecv; i += 3) {
          double xPt = xyzRecvVals[i];
          double yPt = xyzRecvVals[i + 1];
          double zPt = xyzRecvVals[i + 2];

          if( (xPt >= 0.0) && (yPt >= 0.0) && (zPt >= 0.0) 
              && (xPt < 1.0) && (yPt < 1.0) && (zPt < 1.0) ) {
            unsigned int ei = static_cast<int>(xPt*static_cast<double>(Ne));
            unsigned int ej = static_cast<int>(yPt*static_cast<double>(Ne));
            unsigned int ek = static_cast<int>(zPt*static_cast<double>(Ne));

            double x0 = static_cast<double>(ei)/static_cast<double>(Ne);
            double y0 = static_cast<double>(ej)/static_cast<double>(Ne);
            double z0 = static_cast<double>(ek)/static_cast<double>(Ne);

            double psi = ((xPt - x0)*2.0*static_cast<double>(Ne)) - 1.0;
            double eta = ((yPt - y0)*2.0*static_cast<double>(Ne)) - 1.0;
            double gamma = ((zPt - z0)*2.0*static_cast<double>(Ne)) - 1.0;
            double phiVals[8][4];

            evalAll3Dcubic(psi, eta, gamma, phiVals);

            for(int d = 0; d < numImages; d++) {
              //interpolate tau 
              double tauAtUval = 0;

              for(int j = 0; j < 8; j++) {
                int xid = ei + (j%2);
                int yid = ej + ((j/2)%2);
                int zid = ek + (j/4);
                tauAtUval += ((tauGhostedArr[zid][yid][xid][d])*(phiVals[j][0]));
                for(int k = 0; k < 3; k++) {
                  tauAtUval += ((0.5/static_cast<double>(Ne))*
                      (gradTauGhostedArr[zid][yid][xid][(3*d) + k])*(phiVals[j][1 + k]));
                }//end for k
              }//end for j

              sendTauVals[((i/3)*numImages) + d] = tauAtUval;
            }//end for d
          }//end if in domain
        }//end for i

        DAVecRestoreArrayDOF(da1dof[lev], tauGhosted, &tauGhostedArr);
        DAVecRestoreArrayDOF(da3dof, gradTauGhosted, &gradTauGhostedArr);

        VecDestroy(tauVec);
        VecDestroy(gradTauVec);
        VecDestroy(tauGhosted);
        VecDestroy(gradTauGhosted);

        DADestroy(da3dof);

        double* recvTauVals = new double[numImages*(totalSend/3)];
        assert(recvTauVals);

        for(int i = 0; i < npesCurr[lev]; i++) {
          sendCnts[i] = (numImages*(sendCnts[i]/3));
          sendOffsets[i] = (numImages*(sendOffsets[i]/3));
          recvCnts[i] = (numImages*(recvCnts[i]/3));
          recvOffsets[i] = (numImages*(recvOffsets[i]/3));
        }//end for i

        par::Mpi_Alltoallv_sparse<double>(sendTauVals, recvCnts, recvOffsets, 
            recvTauVals, sendCnts, sendOffsets, commCurr[lev]);

        for(int i = 0; i < npesCurr[lev]; i++) {
          tmpSendCnts[i] = 0;
        }//end for i

        for(int zi = zs; zi < (zs + nze); zi++) {
          for(int yi = ys; yi < (ys + nye); yi++) {
            for(int xi = xs; xi < (xs + nxe); xi++) {
              int idx = ((((zi - zs)*nye) + (yi - ys))*nxe) + (xi - xs);
              int sendId = sendOffsets[part[idx]] + tmpSendCnts[part[idx]];
              tmpSendCnts[part[idx]] += numImages;
              for(int d = 0; d < numImages; d++) {
                tauElemental[lev][(idx*numImages) + d] = recvTauVals[sendId + d];
              }//end for d
            }//end for xi
          }//end for yi
        }//end for zi

        assert(sendTauVals);
        delete [] sendTauVals;
        sendTauVals = NULL;

        assert(recvTauVals);
        delete [] recvTauVals;
        recvTauVals = NULL;

        assert(xyzSendVals);
        delete [] xyzSendVals;
        xyzSendVals = NULL;

        assert(xyzRecvVals);
        delete [] xyzRecvVals;
        xyzRecvVals = NULL;

        assert(sendOffsets);
        delete [] sendOffsets;
        sendOffsets = NULL;

        assert(recvOffsets);
        delete [] recvOffsets;
        recvOffsets = NULL;

        assert(tmpSendCnts);
        delete [] tmpSendCnts;
        tmpSendCnts = NULL;

        assert(sendCnts);
        delete [] sendCnts;
        sendCnts = NULL;

        assert(recvCnts);
        delete [] recvCnts;
        recvCnts = NULL;

        assert(part);
        delete [] part;
        part = NULL;

      }//end if active
    }//end if coarsest

    rgNodeVals.clear();

    PetscLogEventEnd(tauElemAtUEvent, 0, 0, 0, 0);

    if(rank < npesCurr[lev]) {
      //Image to octree
      std::vector<ot::TreeNode> linOct;
      for(int imgId = 0; imgId < numImages; imgId++) {
        std::vector<double> tmpSigElemental;
        std::vector<double> tmpTauElemental;
        for(int i = imgId; i < sigElemental[lev].size(); i += numImages) {
          tmpSigElemental.push_back(sigElemental[lev][i]);
        }//end for i
        for(int i = imgId; i < tauElemental[lev].size(); i += numImages) {
          tmpTauElemental.push_back(tauElemental[lev][i]);
        }//end for i

        std::vector<ot::TreeNode> linSigOct;
        std::vector<ot::TreeNode> linTauOct;
        ot::regularGrid2Octree(tmpSigElemental, Ne, nxe, nye, nze, xs, ys, zs,
            linSigOct, dim, maxDepth, threshold, commCurr[lev]);
        ot::regularGrid2Octree(tmpTauElemental, Ne, nxe, nye, nze, xs, ys, zs,
            linTauOct, dim, maxDepth, threshold, commCurr[lev]);

        tmpSigElemental.clear();
        tmpTauElemental.clear();

        unsigned int locSigSz = linSigOct.size();
        unsigned int locTauSz = linTauOct.size();
        unsigned int globSigSz;
        unsigned int globTauSz;
        par::Mpi_Allreduce<unsigned int>(&locSigSz, &globSigSz, 1, MPI_SUM, commCurr[lev]);
        par::Mpi_Allreduce<unsigned int>(&locTauSz, &globTauSz, 1, MPI_SUM, commCurr[lev]);

        if(!rank) {
          std::cout<<"Lev: "<<lev<<"ImgId: "<<imgId<<" globSigSz: "<<globSigSz<<" globTauSz: "<<globTauSz<<std::endl;
        }

        std::vector<ot::TreeNode> newLinOct;
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
            ot::mergeOctrees(linTauOct, tmpLinOct, newLinOct, tmpComm);
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
            ot::mergeOctrees(linSigOct, tmpLinOct, newLinOct, tmpComm);
          } else {
            std::vector<ot::TreeNode> tmpLinOct;
            par::scatterValues<ot::TreeNode>(linTauOct, tmpLinOct, 0, commCurr[lev]);
          }
        }

        linSigOct.clear();
        linTauOct.clear();

        std::vector<ot::TreeNode> oldLinOct = linOct;

        unsigned int locOldSz = oldLinOct.size();
        unsigned int locNewSz = newLinOct.size();
        unsigned int globOldSz;
        unsigned int globNewSz;
        par::Mpi_Allreduce<unsigned int>(&locOldSz, &globOldSz, 1, MPI_SUM, commCurr[lev]);
        par::Mpi_Allreduce<unsigned int>(&locNewSz, &globNewSz, 1, MPI_SUM, commCurr[lev]);

        if(!rank) {
          std::cout<<"Lev: "<<lev<<"ImgId: "<<imgId<<" globOldSz: "<<globOldSz<<" globNewSz: "<<globNewSz<<std::endl;
        }

        if(globNewSz > globOldSz) {
          MPI_Comm tmpComm;
          par::splitComm2way(newLinOct.empty(), &tmpComm, commCurr[lev]);
          if(!(newLinOct.empty())) {
            int tmpNpes, tmpRank;
            MPI_Comm_size(tmpComm, &tmpNpes);
            MPI_Comm_rank(tmpComm, &tmpRank);
            unsigned int avgSz = globOldSz/tmpNpes;
            unsigned int extra = globOldSz%tmpNpes;
            std::vector<ot::TreeNode> tmpLinOct;
            if(tmpRank < extra) {
              par::scatterValues<ot::TreeNode>(oldLinOct, tmpLinOct, (avgSz + 1), commCurr[lev]);
            } else {
              par::scatterValues<ot::TreeNode>(oldLinOct, tmpLinOct, avgSz, commCurr[lev]);
            }
            ot::mergeOctrees(newLinOct, tmpLinOct, linOct, tmpComm);
          } else {
            std::vector<ot::TreeNode> tmpLinOct;
            par::scatterValues<ot::TreeNode>(oldLinOct, tmpLinOct, 0, commCurr[lev]);
          }
        } else {
          MPI_Comm tmpComm;
          par::splitComm2way(oldLinOct.empty(), &tmpComm, commCurr[lev]);
          if(!(oldLinOct.empty())) {
            int tmpNpes, tmpRank;
            MPI_Comm_size(tmpComm, &tmpNpes);
            MPI_Comm_rank(tmpComm, &tmpRank);
            unsigned int avgSz = globNewSz/tmpNpes;
            unsigned int extra = globNewSz%tmpNpes;
            std::vector<ot::TreeNode> tmpLinOct;
            if(tmpRank < extra) {
              par::scatterValues<ot::TreeNode>(newLinOct, tmpLinOct, (avgSz + 1), commCurr[lev]);
            } else {
              par::scatterValues<ot::TreeNode>(newLinOct, tmpLinOct, avgSz, commCurr[lev]);
            }
            ot::mergeOctrees(oldLinOct, tmpLinOct, linOct, tmpComm);
          } else {
            std::vector<ot::TreeNode> tmpLinOct;
            par::scatterValues<ot::TreeNode>(newLinOct, tmpLinOct, 0, commCurr[lev]);
          }
        }

      }//end for imgId

      sigElemental[lev].clear();
      tauElemental[lev].clear();

      std::vector<ot::TreeNode> balOct;
      ot::balanceOctree(linOct, balOct, dim, maxDepth, incCorner, commCurr[lev]);
      linOct.clear();

      int nlevels = 1; 
      PetscInt nlevelsPetscInt = nlevels;
      PetscOptionsGetInt(0, "-nlevels", &nlevelsPetscInt, 0);
      nlevels = nlevelsPetscInt;

      bool balOctEmpty = balOct.empty();

      MPI_Comm balOctComm;
      par::splitComm2way(balOctEmpty, &balOctComm, commCurr[lev]);

      if(!balOctEmpty) {
        int balOctNpes;
        MPI_Comm_size(balOctComm, &balOctNpes);
        assert(rank < balOctNpes);
        ot::DAMGCreateAndSetDA(balOctComm, nlevels, NULL, &damg, 
            balOct, dof, mgLoadFac, compressLut, incCorner);
      }
      balOct.clear();

      std::vector<std::vector<double> > sigLocal;
      std::vector<std::vector<double> > gradSigLocal;
      std::vector<std::vector<double> > tauLocal;
      std::vector<std::vector<double> > gradTauLocal;

      ot::DA* thisDA = NULL;

      if(damg) {
        ot::PrintDAMG(damg);
        ot::DAMGCreateSuppressedDOFs(damg);
        thisDA = DAMGGetDA(damg);
      }

      createImagePatches(da1dof[lev], thisDA, padding, numImages,
          sigGlobal[lev], tauGlobal[lev], gradSigGlobal[lev], gradTauGlobal[lev],
          sigLocal, tauLocal, gradSigLocal, gradTauLocal);

      DADestroy(da1dof[lev]);

      sigGlobal[lev].clear();
      gradSigGlobal[lev].clear();
      tauGlobal[lev].clear();
      gradTauGlobal[lev].clear();

      if(damg) {
        createHessContexts(damg, Ne, padding, numImages,
            sigLocal, gradSigLocal, tauLocal, gradTauLocal,
            PhiMatStencil, LaplacianStencil, GradDivStencil, 
            numGpts, gWts, gPts, mu, lambda, alpha);
      }

      sigLocal.clear();
      gradSigLocal.clear();
      tauLocal.clear();
      gradTauLocal.clear();

    }//end if active

    ot::DA* dao = NULL;
    std::vector<double> nodePts;
    if(damg) {
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

      int damgPrevNpesActive;

      if(!rank) {
        assert(damgPrev != NULL);
        assert((DAMGGetDA(damgPrev)) != NULL);
        damgPrevNpesActive = (DAMGGetDA(damgPrev))->getNpesActive();
      }

      par::Mpi_Bcast<int>(&damgPrevNpesActive, 1, 0, commAll);

      int newAvgSize = numGlobalPts/damgPrevNpesActive;
      int newExtra = numGlobalPts%damgPrevNpesActive;

      std::vector<double> nodePtsDup;
      if( rank >= damgPrevNpesActive) {
        par::scatterValues<double>(nodePts, nodePtsDup, 0, commAll);
      } else if (rank < newExtra) {
        par::scatterValues<double>(nodePts, nodePtsDup, (3*(newAvgSize + 1)), commAll);
      } else {
        par::scatterValues<double>(nodePts, nodePtsDup, (3*newAvgSize), commAll);
      }
      nodePts.clear();

      std::vector<double> nodeValsDup;
      if(damgPrev) {
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

    if(damg) {
      Vec Uin;
      dao->createVector(Uin, false, false, 3);
      VecZeroEntries(Uin);

      if(lev) {
        int ptsCtr = 0;
        if(dao->iAmActive()) {
          PetscScalar* inArr;
          dao->vecGetBuffer(Uin, inArr, false, false, false, 3);

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

          dao->vecRestoreBuffer(Uin, inArr, false, false, false, 3);
        }//end if active
      }//end if coarsest
      nodeVals.clear();

      Vec Uout;
      VecDuplicate(Uin, &Uout);

      if(!rank) {
        std::cout<<"Starting GaussNewton..."<<std::endl;
      }

      objImgValOut = gaussNewton(damg, fTol, xTol, maxIterCnt, Uin, Uout);

    	if(!rank) {
		    std::cout << "Objective Image Value: " << objImgValOut << std::endl;
      }
 
      destroyHessContexts(damg);

      VecDestroy(Uin);

      PetscScalar* octDispVals;
      VecGetArray(Uout, &octDispVals);

      PetscInt dispOctSz;
      VecGetLocalSize(Uout, &dispOctSz);

      dispOct.resize(dispOctSz);
      for(int i = 0; i < dispOctSz; i++) {
        dispOct[i] = octDispVals[i];
      }//end for i

      VecRestoreArray(Uout, &octDispVals);

      VecDestroy(Uout);
    }//end if active

  }//end for lev

  assert(commCurr);
  delete [] commCurr;
  commCurr = NULL;

  assert(npesCurr);
  delete [] npesCurr;
  npesCurr = NULL;

  assert(sigElemental);
  delete [] sigElemental;
  sigElemental = NULL;

  assert(tauElemental);
  delete [] tauElemental;
  tauElemental = NULL;

  assert(sigGlobal);
  delete [] sigGlobal;
  sigGlobal = NULL;

  assert(tauGlobal);
  delete [] tauGlobal;
  tauGlobal = NULL;

  assert(gradSigGlobal);
  delete [] gradSigGlobal;
  gradSigGlobal = NULL;

  assert(gradTauGlobal);
  delete [] gradTauGlobal;
  gradTauGlobal = NULL;

  assert(da1dof);
  delete [] da1dof;
  da1dof = NULL;

  destroyLmat(LaplacianStencil);
  destroyGDmat(GradDivStencil);
  destroyPhimat(PhiMatStencil, numGpts);

  destroyGaussPtsAndWts(gPts, gWts);

  DA dar;
  Vec UrgGlobal;

  if(damgPrev) {
    DACreate3d(damgPrev[0]->comm, DA_NONPERIODIC, DA_STENCIL_BOX, Nfe + 1, Nfe + 1, Nfe + 1,
        PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
        3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dar);

    mortonToRgGlobalDisp(damgPrev, dar, Nfe, dispOct, UrgGlobal);
  }//end if active

  dispOct.clear();

  PetscTruth saveFinalRgDisp;
  PetscOptionsHasName(0, "-saveFinalRgDisp", &saveFinalRgDisp);

  if(damgPrev) {

    if(saveFinalRgDisp) {
      Vec UrgNatural;
      DACreateNaturalVector(dar, &UrgNatural);

      DAGlobalToNaturalBegin(dar, UrgGlobal, INSERT_VALUES, UrgNatural);
      DAGlobalToNaturalEnd(dar, UrgGlobal, INSERT_VALUES, UrgNatural);

      char fname[256];
      sprintf(fname, "%s_%d_%d.dat", disp, rank, npesAll);
      saveVector(UrgNatural, fname);

      VecDestroy(UrgNatural);
    }

    double maxDetJac, minDetJac;
    detJacMaxAndMin(dar, UrgGlobal, &maxDetJac, &minDetJac);
    if(!rank) {
      std::cout<<" Max Det Jac: "<<maxDetJac<<" Min Det Jac: "<<minDetJac<<std::endl;
		  std::cout << "Final objective Image Value: " << objImgValOut << std::endl;

    }

    VecDestroy(UrgGlobal);

    DADestroy(dar);

    ot::DAMGDestroy(damgPrev);
    damgPrev = NULL;

  }//end if active

  if (!rank){
      FILE *objFunc= fopen(obj,"w");
      fprintf(objFunc,"%1.4e",objImgValOut);
      fclose(objFunc);
  }


  ot::DAMG_Finalize();
  return objImgValOut;

}


double iterRunOpt(char *targetImg, char *templateImg, char *disp, char *obj, int numIter){
  char curDispImg[100];
  char nextDispImg[100];
  char elasDisp[100];
  char genDisp[100];
  char objFunc[100];

    strcpy(curDispImg,templateImg);
    strcpy(elasDisp,disp);
    strcpy(objFunc,obj);

    PetscInt numImages=1;
    PetscOptionsGetInt(0,"-numImages",&numImages,0);

    double imgObj=0;

    for (int iterNum=0;iterNum < numIter;iterNum++){
        //call registration
        printf("At iternum=%d, target img: %s, templateImg: %s, disp: %s, objective function: %s\n",iterNum,targetImg,curDispImg,elasDisp,objFunc);
        imgObj = elasRunOpt(targetImg,curDispImg,elasDisp,objFunc);
        //create displaced image
        sprintf(nextDispImg,"%s.%d",templateImg,iterNum);
        sprintf(genDisp,"%s_0_1.dat",elasDisp);
        sprintf(objFunc,"%s.%d",obj,iterNum);

        printf("The current displaced image %s \n",curDispImg);
        printf("The next displaced image %s \n",nextDispImg);
        printf("The displacedout for genImg %s \n",genDisp);
        //options for genImg: 
        //displacement scale is 1
        //computeDetJac is enabled
        double dispScale=1.0;
        int computeDetJac=1;
        genDispImg(genDisp,dispScale,curDispImg,nextDispImg,computeDetJac,numImages);
        strcpy(curDispImg,nextDispImg);
        sprintf(elasDisp,"%s.%d",disp,iterNum);
 
   }

  return imgObj;
}


double iterRunOptPar(char *tgtImg,char *tmplImg, char *dispFile, char *objFile, int numIter){

    char targetImg[100]; 
    char curDispImg[100];
    char nextDispImg[100];
    char elasDisp[100];
    char genDisp[100];
    char objFunc[100];

    strcpy(targetImg,tgtImg); 
    strcpy(curDispImg,tmplImg);
    strcpy(elasDisp,dispFile);
    strcpy(objFunc,objFile);


    PetscInt numImages=1;
    PetscOptionsGetInt(0,"-numImages",&numImages,0);

    double imgObj=0;
    double regTime;
    double totalTime=0; 
    MPI_Comm commAll = MPI_COMM_WORLD;

    int rank, npesAll;
    MPI_Comm_rank(commAll, &rank);
    MPI_Comm_size(commAll, &npesAll);

    for (int iterNum=0;iterNum < numIter;iterNum++){
        if (!rank){
        printf("\nAt iternum=%d, target img: %s, templateImg: %s, disp: %s, objective function: %s\n",iterNum,tgtImg,curDispImg,elasDisp,objFunc);
        }
        //call parallel registration
         regTime = -get_seconds();
         elasRunOpt(targetImg,curDispImg,elasDisp,objFunc);

        //create displaced image
        sprintf(nextDispImg,"%s.%d",tmplImg,iterNum);
        sprintf(genDisp,"%s.dat",elasDisp);
        sprintf(objFunc,"%s.%d",objFile,iterNum);

         if (!rank){
            printf("The current displaced image %s \n",curDispImg);
            printf("The next displaced image %s \n",nextDispImg);
            printf("The displacedout for genImg %s \n",genDisp);
            
            concatDispFile(elasDisp,npesAll,genDisp);
            //options for genImg: 
            //displacement scale is 1
            //computeDetJac is enabled
            double dispScale=1.0;
            int computeDetJac=1;
            int numImg=1;
            genDispImg(genDisp,dispScale,curDispImg,nextDispImg,computeDetJac,numImg);
        }
        regTime += get_seconds();
        totalTime += regTime;
        MPI_Barrier(commAll);
        strcpy(curDispImg,nextDispImg);

        sprintf(elasDisp,"%s.%d",dispFile,iterNum);

   }

   if (!rank) fprintf(stdout,"Total time is %f\n",totalTime);

}

double iterVecTest(char *tgtImg,char *tmplImg, char *dispFile, char *objFile, int numIter){

    char targetImg[100]; 
    char curDispImg[100];
    char nextDispImg[100];
    char elasDisp[100];
    char genDisp[100];
    char objFunc[100];

    strcpy(targetImg,tgtImg); 
    strcpy(curDispImg,tmplImg);
    strcpy(elasDisp,dispFile);
    strcpy(objFunc,objFile);


    PetscInt numImages=1;
    PetscOptionsGetInt(0,"-numImages",&numImages,0);

    double imgObj=0;
    double vecRegTime;  
    double totalTime=0; 
    MPI_Comm commAll = MPI_COMM_WORLD;

    int rank, npesAll;
    MPI_Comm_rank(commAll, &rank);
    MPI_Comm_size(commAll, &npesAll);

    for (int iterNum=0;iterNum < numIter;iterNum++){
        if (!rank){
        printf("\nAt iternum=%d, target img: %s, templateImg: %s, disp: %s, objective function: %s\n",iterNum,tgtImg,curDispImg,elasDisp,objFunc);
        }
        //call parallel registration
         vecRegTime = -get_seconds();
//         elasVecRunOpt(targetImg,curDispImg,elasDisp,objFunc);

        sprintf(nextDispImg,"%s.%d",tmplImg,iterNum);
        sprintf(genDisp,"%s.dat",elasDisp);

         if (!rank){
            printf("The current displaced image %s \n",curDispImg);
            printf("The next displaced image %s \n",nextDispImg);
            printf("The displacedout for genImg %s \n",genDisp);
            
//            concatDispFile(elasDisp,npesAll,genDisp);
            //options for genImg: 
            //displacement scale is 1
            //computeDetJac is enabled
            double dispScale=1.0;
            int computeDetJac=1;
            
            genDispImg(genDisp,dispScale,curDispImg,nextDispImg,computeDetJac,numImages);
        }



   }
    if (!rank) fprintf(stdout,"Total time is %f\n",totalTime);



}

double iterVecRunOptPar(char *tgtImg,char *tmplImg, char *dispFile, char *objFile, int numIter){

    char targetImg[100]; 
    char curDispImg[100];
    char nextDispImg[100];
    char elasDisp[100];
    char genDisp[100];
    char objFunc[100];

    strcpy(targetImg,tgtImg); 
    strcpy(curDispImg,tmplImg);
    strcpy(elasDisp,dispFile);
    strcpy(objFunc,objFile);


    PetscInt numImages=1;
    PetscOptionsGetInt(0,"-numImages",&numImages,0);

    double imgObj=0;
    double vecRegTime;  
    double totalTime=0; 
    MPI_Comm commAll = MPI_COMM_WORLD;

    int rank, npesAll;
    MPI_Comm_rank(commAll, &rank);
    MPI_Comm_size(commAll, &npesAll);

    for (int iterNum=0;iterNum < numIter;iterNum++){
        if (!rank){
        printf("\nAt iternum=%d, target img: %s, templateImg: %s, disp: %s, objective function: %s\n",iterNum,tgtImg,curDispImg,elasDisp,objFunc);
        }
        //call parallel registration
         vecRegTime = -get_seconds();
         elasVecRunOpt(targetImg,curDispImg,elasDisp,objFunc);

        //create displaced image
        sprintf(nextDispImg,"%s.%d",tmplImg,iterNum);
        sprintf(genDisp,"%s.dat",elasDisp);
        sprintf(objFunc,"%s.%d",objFile,iterNum);

         if (!rank){
            printf("The current displaced image %s \n",curDispImg);
            printf("The next displaced image %s \n",nextDispImg);
            printf("The displacedout for genImg %s \n",genDisp);
            
            concatDispFile(elasDisp,npesAll,genDisp);
            //options for genImg: 
            //displacement scale is 1
            //computeDetJac is enabled
            double dispScale=1.0;
            int computeDetJac=1;
            
            genDispImg(genDisp,dispScale,curDispImg,nextDispImg,computeDetJac,numImages);
        }

        MPI_Barrier(commAll);
        vecRegTime += get_seconds();
        totalTime += vecRegTime;
        printf("At iterNum=%d, the time is %.4e\n",vecRegTime);
        strcpy(curDispImg,nextDispImg);

        sprintf(elasDisp,"%s.%d",dispFile,iterNum);

   }
    if (!rank) fprintf(stdout,"Total time is %f\n",totalTime);


}
