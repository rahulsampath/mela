
#include "mpi.h"
#include "sys.h"
#include "omg.h"
#include "registration.h"
#include "regInterpCubic.h"
#include "externVars.h"
#include "dendro.h"

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

  ot::DAMG_Initialize(MPI_COMM_WORLD);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ot::DAMG *damg;    
  unsigned int dim = 3;
  PetscInt maxDepth = 30;
  int nlevels = 1; 
  PetscInt Ne = 128;
  unsigned int dof = 3;  
  PetscReal mu = 1.0;
  PetscReal lambda = 4.0;
  bool incCorner = 1;  
  PetscTruth compressLut;
  PetscReal mgLoadFac = 2.0;

  PetscOptionsHasName(0, "-compressLut", &compressLut);

  if(compressLut) {
    if(!rank) {
      std::cout<<"Mesh is Compressed."<<std::endl;
    }
  }

  //Functions for using KSP_Shell (will be used @ the coarsest grid if not all
  //processors are active on the coarsest grid)

  ot::getPrivateMatricesForKSP_Shell = getPrivateMatricesForKSP_Shell_Hess;

  //Set function pointers so that PC_BlockDiag could be used.

  ot::getDofAndNodeSizeForPC_BlockDiag = getDofAndNodeSizeForHessMat;

  ot::computeInvBlockDiagEntriesForPC_BlockDiag = computeInvBlockDiagEntriesForHessMat;

  PetscInt nlevelsPetscInt = nlevels;
  PetscOptionsGetInt(0, "-nlevels", &nlevelsPetscInt, 0);
  nlevels = nlevelsPetscInt;

  PetscOptionsGetInt(0, "-Ne", &Ne, 0);
  PetscOptionsGetInt(0, "-maxDepth", &maxDepth, 0);

  PetscOptionsGetReal(0, "-mu", &mu, 0);
  PetscOptionsGetReal(0, "-lambda", &lambda, 0);
  PetscOptionsGetReal(0, "-mgLoadFac", &mgLoadFac, 0);

  std::vector<std::vector<double> > tau;
  std::vector<std::vector<double> > gradTau;
  tau.resize(1);
  gradTau.resize(1);
  tau[0].resize((Ne + 1)*(Ne + 1)*(Ne + 1));
  gradTau[0].resize(3*(Ne + 1)*(Ne + 1)*(Ne + 1));

  for(int i = 0; i < ((Ne + 1)*(Ne + 1)*(Ne + 1)); i++) {
    tau[0][i] = 0.0;
    for(int j = 0; j < 3; j++) {
      gradTau[0][(3*i) + j] = 0.0;
    }
  }

  unsigned int regLev = binOp::fastLog2(Ne);

  std::vector<ot::TreeNode> balOct;
  ot::createRegularOctree(balOct, regLev, 3, maxDepth, MPI_COMM_WORLD);

  ot::DAMGCreateAndSetDA(MPI_COMM_WORLD, nlevels, NULL, &damg, 
      balOct, dof, mgLoadFac, compressLut, incCorner);

  ot::PrintDAMG(damg);

  double**** LaplacianStencil; 
  double**** GradDivStencil;
  double****** PhiMatStencil; 
  
  int numGpts = 4;
  double gWts[] = {0.65214515, 0.65214515, 0.34785485, 0.34785485};
  double gPts[] = {0.33998104, -0.33998104, 0.86113631, -0.86113631};

  /*
  int numGpts = 3;
  double gWts[] = {0.88888889, 0.555555556, 0.555555556};
  double gPts[] = {0.0, 0.77459667, -0.77459667};
  */
  
  createLmat(LaplacianStencil);
  createGDmat(GradDivStencil);
  createPhimat(PhiMatStencil, numGpts, gPts);

  ot::DAMGCreateSuppressedDOFs(damg);

  //It should really be tau and gradTau, but this is only for testing
  createHessContexts(damg, Ne, tau, gradTau, tau, gradTau,
      PhiMatStencil, LaplacianStencil, GradDivStencil,
      numGpts, gWts, gPts, mu, lambda, 1.0);

  HessData* ctx = static_cast<HessData*>(damg[nlevels - 1]->user);

  ot::DA* dao = damg[nlevels - 1]->da;

  Vec U;
  dao->createVector(U, false, false, 3);
  VecZeroEntries(U);

  updateHessContexts(damg, U);

  ot::DAMGSetKSP(damg, createHessMat,  computeHessMat, computeGaussianRHS);

  ot::DAMGSolve(damg);

  Vec sol = DAMGGetx(damg);

  enforceBC(dao, ctx->bdyArr, sol);

  //Convert from Morton ordering to DA ordering
  DA dar;

  DACreate3d(MPI_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dar);

  Vec solRgNatural, solRgGlobal;
  DACreateNaturalVector(dar, &solRgNatural);
  DACreateGlobalVector(dar, &solRgGlobal);
  VecZeroEntries(solRgNatural);

  PetscScalar* solArr;
  dao->vecGetBuffer(sol, solArr, false, false, true, 3);

  double h = 1.0/static_cast<double>(Ne);

  if(dao->iAmActive()) {
    unsigned int maxD = dao->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);

    //No communication. Each processor sets the values for the nodes it owns.
    //We don't care for the positive boundaries since they will be zero anyway

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
        int ei = static_cast<int>(x0/h);
        int ej = static_cast<int>(y0/h);
        int ek = static_cast<int>(z0/h);

        int nodeId = (((ek*(Ne + 1)) + ej)*(Ne + 1)) + ei;
        for(int i = 0; i < 3; i++) {
          VecSetValue(solRgNatural, ((3*nodeId) + i), solArr[(3*idx) + i], INSERT_VALUES);
        }
      }
    }//end WRITABLE
  }//end if active

  dao->vecRestoreBuffer(sol, solArr, false, false, true, 3);

  VecAssemblyBegin(solRgNatural);
  VecAssemblyEnd(solRgNatural);

  DANaturalToGlobalBegin(dar, solRgNatural, INSERT_VALUES, solRgGlobal);
  DANaturalToGlobalEnd(dar, solRgNatural, INSERT_VALUES, solRgGlobal);

  bool repeatLoop = true;
  double scaleFac = 1.0;
  double maxDetJac, minDetJac;
  while(repeatLoop) {
    if(!rank) {
      std::cout<<"Current Scale Factor: "<<scaleFac<<std::endl;
    }
    detJacMaxAndMin(dar, solRgGlobal, &maxDetJac, &minDetJac);
    if( (minDetJac > 0.01) && (maxDetJac < 10) ) {
      repeatLoop = false;
    } else {
      VecScale(solRgGlobal, 0.9);
      scaleFac *= 0.9;
    }
  }

  if(!rank) {
    std::cout<<"Final Scale Fac = "<<scaleFac<<std::endl;
  }

  VecScale(solRgNatural, scaleFac);

  PetscReal solNorm;
  VecNorm(solRgNatural, NORM_2, &solNorm);

  Vec solRgNaturalZero;
  VecScatter toZero;

  VecScatterCreateToZero(solRgNatural, &toZero, &solRgNaturalZero);

  VecScatterBegin(toZero, solRgNatural, solRgNaturalZero, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(toZero, solRgNatural, solRgNaturalZero, INSERT_VALUES, SCATTER_FORWARD);

  if(!rank) {
    saveVector(solRgNaturalZero, "disp.dat");
    std::cout<<"Final DetJac max: "<<maxDetJac<<" min: "<<minDetJac<<std::endl;
    std::cout<<" norm-2 of solution: "<<solNorm<<std::endl;
  }

  DADestroy(dar);

  VecDestroy(U);
  VecDestroy(solRgNatural);
  VecDestroy(solRgGlobal);
  VecDestroy(solRgNaturalZero);
  VecScatterDestroy(toZero);

  destroyLmat(LaplacianStencil);
  destroyGDmat(GradDivStencil);
  destroyPhimat(PhiMatStencil, numGpts);

  destroyHessContexts(damg);

  ot::DAMGDestroy(damg);

  ot::DAMG_Finalize();

  PetscFinalize();

}



