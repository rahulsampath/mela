
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

int tauElemAtUEvent;

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, "options", 0);

  ot::RegisterEvents();

  MPI_Comm commAll = MPI_COMM_WORLD;

  int rank, npesAll;
  MPI_Comm_rank(commAll, &rank);
  MPI_Comm_size(commAll, &npesAll);

  if(argc < 4) {
    if(!rank) {
      std::cout<<"Usage: exe fixedImg movingImg DispFile"<<std::endl;
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

  std::vector<std::vector<double> >* sigGlobal = new std::vector<std::vector<double> >[numOptLevels];
  assert(sigGlobal);

  std::vector<std::vector<double> >* gradSigGlobal = new std::vector<std::vector<double> >[numOptLevels];
  assert(gradSigGlobal);

  std::vector<std::vector<double> >* tauGlobal = new std::vector<std::vector<double> >[numOptLevels];
  assert(tauGlobal);

  std::vector<std::vector<double> >* gradTauGlobal = new std::vector<std::vector<double> >[numOptLevels];
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


  //New Block
  {
    int lev = 4;
    int Ne = Nce*(1u << lev);
    double fTol = fTolInit/(static_cast<double>(Ne));
    double xTol = xTolInit/(static_cast<double>(Ne));
    if(!patchWidthSetFromOption) {
      imgPatchWidth = patchWidthFac/static_cast<double>(Ne);
    }

    if(!rank) {
      std::cout<<"Starting Block..."<<std::endl;
    }

    ot::DAMG *damg = NULL;    

    if(rank < npesCurr[lev]) {
      std::vector<ot::TreeNode> balOct;
      char debugFname[256];
      sprintf(debugFname, "debugOct%d.ot", rank);
      readNodesFromFile(debugFname, balOct);

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

      std::vector<ot::TreeNode> imgPatches;
      std::vector<unsigned int> mesh;

      if(damg) {
        ot::PrintDAMG(damg);
        ot::DAMGCreateSuppressedDOFs(damg);
        createImagePatches(Ne, DAMGGetDA(damg), imgPatches);
        expandImagePatches(Ne, imgPatchWidth, ((DAMGGetDA(damg))->getBlocks()), imgPatches);
        meshImagePatches(imgPatches, mesh);
      }

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

      if(damg) {
        createNewHessContexts(damg, imgPatchWidth, imgPatches, mesh,
            sigLocal, gradSigLocal, tauLocal, gradTauLocal,
            PhiMatStencil, LaplacianStencil, GradDivStencil,
            numGpts, gWts, gPts, mu, lambda, alpha);
      }

      sigLocal.clear();
      gradSigLocal.clear();
      tauLocal.clear();
      gradTauLocal.clear();
      imgPatches.clear();
      mesh.clear();
    }//end if active

    if(!rank) {
      std::cout<<"Loading Uin..."<<std::endl;
    }

    if(damg) {
      ot::DA* dao = DAMGGetDA(damg);

      Vec Uin;
      dao->createVector(Uin, false, false, 3);
      VecZeroEntries(Uin);

      if(dao->iAmActive()) {
        char uinFname[256];
        sprintf(uinFname, "uinVec%d.dat", rank);
        FILE* fptr = fopen(uinFname, "rb");

        PetscInt vlen;
        fread(&vlen, sizeof(PetscInt), 1, fptr);

        PetscScalar* inArr;      
        VecGetArray(Uin, &inArr);

        fread(inArr, sizeof(PetscScalar), vlen, fptr);

        VecRestoreArray(Uin, &inArr);

        fclose(fptr);
      }

      Vec Uout;
      VecDuplicate(Uin, &Uout);

      if(!rank) {
        std::cout<<"Starting GaussNewton..."<<std::endl;
      }

      newGaussNewton(damg, fTol, xTol, maxIterCnt, imgPatchWidth, Uin, Uout);

      destroyHessContexts(damg);

      VecDestroy(Uin);
      VecDestroy(Uout);

      ot::DAMGDestroy(damg);
    }//end if active

  }//end new block

  if(!rank) {
    std::cout<<"Cleaning up..."<<std::endl;
  }

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

  assert(gPts);
  delete [] gPts;
  gPts = NULL;

  assert(gWts);
  delete [] gWts;
  gWts = NULL;

  ot::DAMG_Finalize();

  PetscFinalize();

}



