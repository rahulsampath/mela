
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
  PetscInitialize(&argc, &argv, "optionsTest", 0);
  ot::RegisterEvents();
  ot::DAMG_Initialize(MPI_COMM_WORLD);

  int rank, npes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  //Elemental tau vector on processor 0.
  Vec tauN0;

  VecCreate(MPI_COMM_WORLD, &tauN0);

  int Ne;

  if(!rank) {
    struct dsr hdrTau;
    std::vector<double> tauImg;

    readImage(argv[1], &hdrTau, tauImg);

    Ne = hdrTau.dime.dim[1];

    assert(hdrTau.dime.dim[2] == Ne);
    assert(hdrTau.dime.dim[3] == Ne);

    VecSetSizes(tauN0, (Ne + 1)*(Ne + 1)*(Ne + 1), PETSC_DECIDE);
    VecSetType(tauN0, VECMPI);

    PetscScalar* tauArr;
    VecGetArray(tauN0, &tauArr);

    for(int k = 0 ; k < Ne; k++) {
      for(int j = 0 ; j < Ne; j++) {
        for(int i = 0 ; i < Ne; i++) {
          int ptIdxN = (((k*(Ne + 1)) + j)*(Ne + 1)) + i;
          int ptIdxE = (((k*Ne) + j)*Ne) + i;
          tauArr[ptIdxN] = tauImg[ptIdxE];
        }
      }
    }

    for(int k = 0 ; k < Ne; k++) {
      for(int j = 0 ; j < Ne; j++) {
        tauArr[(((k*(Ne + 1)) + j)*(Ne + 1)) + Ne] = tauArr[(((k*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1];
      }
    }

    for(int k = 0 ; k < Ne; k++) {
      for(int i = 0 ; i < Ne; i++) {
        tauArr[(((k*(Ne + 1)) + Ne)*(Ne + 1)) + i] = tauArr[(((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i];
      }
    }

    for(int j = 0 ; j < Ne; j++) {
      for(int i = 0 ; i < Ne; i++) {
        tauArr[(((Ne*(Ne + 1)) + j)*(Ne + 1)) + i] = tauArr[((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + i];
      }
    }

    for(int i = 0 ; i < Ne; i++) {
      tauArr[(((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + i] = tauArr[((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + i];
    }

    for(int j = 0 ; j < Ne; j++) {
      tauArr[(((Ne*(Ne + 1)) + j)*(Ne + 1)) + Ne] = tauArr[((((Ne - 1)*(Ne + 1)) + j)*(Ne + 1)) + Ne - 1];
    }

    for(int k = 0 ; k < Ne; k++) {
      tauArr[(((k*(Ne + 1)) + Ne)*(Ne + 1)) + Ne] = tauArr[(((k*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1];
    }

    tauArr[(((Ne*(Ne + 1)) + Ne)*(Ne + 1)) + Ne] = tauArr[((((Ne - 1)*(Ne + 1)) + Ne - 1)*(Ne + 1)) + Ne - 1];

    VecRestoreArray(tauN0, &tauArr);
  } else {
    VecSetSizes(tauN0, 0, PETSC_DECIDE);
    VecSetType(tauN0, VECMPI);
  }//end if p0


  par::Mpi_Bcast<int>(&Ne, 1, 0, MPI_COMM_WORLD);
  double h = 1.0/static_cast<double>(Ne);

  DA da1dof;
  DACreate3d(MPI_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da1dof);

  Vec tauNatural;
  DACreateNaturalVector(da1dof, &tauNatural);

  int* sendSz = NULL;
  int* recvSz = NULL;
  int* sendOff = NULL;
  int* recvOff = NULL;

  PetscInt inSz, outSz;

  VecGetLocalSize(tauN0, &inSz);
  VecGetLocalSize(tauNatural, &outSz);

  //Should actually use MPI_Scatter for optimal performance,
  //but it's okay for now.

  ot::scatterValues(tauN0, tauNatural, inSz, outSz, sendSz, 
      sendOff, recvSz, recvOff, MPI_COMM_WORLD);

  delete [] sendSz;
  delete [] recvSz;
  delete [] sendOff;
  delete [] recvOff;

  VecDestroy(tauN0);

  DA da3dof;
  DACreate3d(MPI_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da3dof);

  Vec tauGlobal;
  Vec tauNaturalAll;
  Vec gradTauGlobal;
  Vec gradTauNatural;
  Vec gradTauNaturalAll;
  VecScatter toLocalAll1dof;
  VecScatter toLocalAll3dof;

  DACreateGlobalVector(da1dof, &tauGlobal);
  DACreateNaturalVector(da3dof, &gradTauNatural);
  DACreateGlobalVector(da3dof, &gradTauGlobal);

  VecScatterCreateToAll(tauNatural, &toLocalAll1dof, &tauNaturalAll);
  VecScatterCreateToAll(gradTauNatural, &toLocalAll3dof, &gradTauNaturalAll);

  DANaturalToGlobalBegin(da1dof, tauNatural, INSERT_VALUES, tauGlobal);
  DANaturalToGlobalEnd(da1dof, tauNatural, INSERT_VALUES, tauGlobal);

  ot::DAMG *damg;    
  unsigned int dim = 3;
  PetscInt maxDepth = 30;
  int nlevels = 1; 
  unsigned int dof = 3;  
  PetscReal mu = 1.0;
  PetscReal lambda = 4.0;
  PetscReal alpha = 1.0;
  PetscReal threshold = 1.0;
  bool incCorner = 1;  
  PetscReal mgLoadFac = 2.0;

  PetscTruth compressLut;
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

  PetscOptionsGetInt(0, "-maxDepth", &maxDepth, 0);
  PetscOptionsGetReal(0, "-mu", &mu, 0);
  PetscOptionsGetReal(0, "-lambda", &lambda, 0);
  PetscOptionsGetReal(0, "-alpha", &alpha, 0);
  PetscOptionsGetReal(0, "-mgLoadFac", &mgLoadFac, 0);
  PetscOptionsGetReal(0, "-threshold", &threshold, 0);

  if(!rank) {
    std::cout<<"alpha = "<<alpha<<" mu = "<<mu<<" lambda = "<<lambda<<std::endl;
  }

  PetscScalar*** tauGlobalArr;
  DAVecGetArray(da1dof, tauGlobal, &tauGlobalArr);

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

  std::vector<double> tauElemental;
  for(int k = zs; k < zs + nze; k++) {
    for(int j = ys; j < ys + nye; j++) {
      for(int i = xs; i < xs + nxe; i++) {
        double xPos = static_cast<double>(i)*h;
        double yPos = static_cast<double>(j)*h;
        double zPos = static_cast<double>(k)*h;
        double valTau = tauGlobalArr[k][j][i];
        tauElemental.push_back(valTau);
      }
    }
  }

  DAVecRestoreArray(da1dof, tauGlobal, &tauGlobalArr);

  computeFDgradient(da1dof, da3dof, tauGlobal, gradTauGlobal);

  DAGlobalToNaturalBegin(da3dof, gradTauGlobal, INSERT_VALUES, gradTauNatural);
  DAGlobalToNaturalEnd(da3dof, gradTauGlobal, INSERT_VALUES, gradTauNatural);

  VecScatterBegin(toLocalAll1dof, tauNatural, tauNaturalAll, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterBegin(toLocalAll3dof, gradTauNatural, gradTauNaturalAll, INSERT_VALUES, SCATTER_FORWARD );

  VecScatterEnd(toLocalAll1dof, tauNatural, tauNaturalAll, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(toLocalAll3dof, gradTauNatural, gradTauNaturalAll, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterDestroy(toLocalAll1dof);
  VecScatterDestroy(toLocalAll3dof);

  VecDestroy(tauNatural);
  VecDestroy(gradTauNatural);

  VecDestroy(tauGlobal);
  VecDestroy(gradTauGlobal);

  DADestroy(da1dof);
  DADestroy(da3dof);

  std::vector<std::vector<double> > tau;
  std::vector<std::vector<double> > gradTau;
  tau.resize(1);
  gradTau.resize(1);
  tau[0].resize((Ne + 1)*(Ne + 1)*(Ne + 1));
  gradTau[0].resize(3*(Ne + 1)*(Ne + 1)*(Ne + 1));

  PetscScalar* tauNaturalAllArr;
  PetscScalar* gradTauNaturalAllArr;

  VecGetArray(tauNaturalAll, &tauNaturalAllArr);
  VecGetArray(gradTauNaturalAll, &gradTauNaturalAllArr);

  for(int i = 0; i < ((Ne + 1)*(Ne + 1)*(Ne + 1)); i++) {
    tau[0][i] = tauNaturalAllArr[i];
    for(int j = 0; j < 3; j++) {
      gradTau[0][(3*i) + j] = gradTauNaturalAllArr[(3*i) + j];
    }
  }

  VecRestoreArray(tauNaturalAll, &tauNaturalAllArr);
  VecRestoreArray(gradTauNaturalAll, &gradTauNaturalAllArr);

  VecDestroy(tauNaturalAll);
  VecDestroy(gradTauNaturalAll);

  PetscTruth usingRegularOctree;
  PetscInt regLev;
  PetscOptionsHasName(0, "-useRegularOctreeAtLevel", &usingRegularOctree);
  PetscOptionsGetInt(0, "-useRegularOctreeAtLevel", &regLev, 0);

  std::vector<ot::TreeNode> balOct;
  if(usingRegularOctree) {
    balOct.clear();
    ot::createRegularOctree(balOct, regLev, 3, maxDepth, MPI_COMM_WORLD);
  } else {
    //Image to octree
    std::vector<ot::TreeNode> linOct;
    ot::regularGrid2Octree(tauElemental, Ne, nxe, nye, nze, xs, ys, zs,
        linOct, dim, maxDepth, threshold, MPI_COMM_WORLD);
    ot::balanceOctree(linOct, balOct, dim, maxDepth, incCorner, MPI_COMM_WORLD);
    linOct.clear();
  }
  tauElemental.clear();

  ot::DAMGCreateAndSetDA(MPI_COMM_WORLD, nlevels, NULL, &damg, 
      balOct, dof, mgLoadFac, compressLut, incCorner);
  balOct.clear();

  ot::PrintDAMG(damg);

  PetscTruth useRandomU;
  PetscOptionsHasName(0, "-useRandomU", &useRandomU);

  PetscTruth saveRandU;
  PetscOptionsHasName(0, "-saveRandU", &saveRandU);

  PetscTruth loadRandU;
  PetscOptionsHasName(0, "-loadRandU", &loadRandU);

  PetscTruth scatterRandU;
  PetscOptionsHasName(0, "-scatterRandU", &scatterRandU);

  Vec U;
  damg[nlevels - 1]->da->createVector(U, false, false, 3);
  if(useRandomU) {
    if(loadRandU) {
      char vecFname[256];
      sprintf(vecFname, "randU_%d_%d.dat",rank,npes);
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
        int* scSendOff = new int[npes];
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
        delete [] scSendOff;
      } else {
        VecCopy(Utmp, U);
      }
      VecDestroy(Utmp);
    } else {
      PetscRandom rctx;  
      PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
      PetscRandomSetType(rctx, PETSCRAND48);
      PetscInt randomSeed = 1652;
      PetscRandomSetSeed(rctx, randomSeed);
      PetscRandomSeed(rctx);
      PetscRandomSetFromOptions(rctx);
      VecSetRandom(U, rctx);
      PetscRandomDestroy(rctx);
      if(saveRandU) {
        char vecFname[256];
        sprintf(vecFname, "randU_%d_%d.dat",rank,npes);
        saveVector(U, vecFname);
      }
    }
  } else {
    VecZeroEntries(U);
  }

  PetscReal uNorm;
  VecNorm(U, NORM_2, &uNorm);

  double**** LaplacianStencil; 
  double**** GradDivStencil;
  double****** PhiMatStencil; 
  int numGpts = 4;
  double gWts[] = {0.65214515, 0.65214515, 0.34785485, 0.34785485};
  double gPts[] = {0.33998104, -0.33998104, 0.86113631, -0.86113631};

  createLmat(LaplacianStencil);
  createGDmat(GradDivStencil);
  createPhimat(PhiMatStencil, numGpts, gPts);

  ot::DAMGCreateSuppressedDOFs(damg);

  //It should really be tau and gradTau, but this is only for testing
  createHessContexts(damg, Ne, tau, gradTau, tau, gradTau,
      PhiMatStencil, LaplacianStencil, GradDivStencil,
      numGpts, gWts, gPts, mu, lambda, alpha);

  HessData* ctx = static_cast<HessData*>(damg[nlevels - 1]->user);

  enforceBC(damg[nlevels - 1]->da, ctx->bdyArr, U);

  updateHessContexts(damg, U);

  ot::DAMGSetKSP(damg, createHessMat,  computeHessMat, computeDummyRHS);

  KSP eigKSP;
  KSPCreate(MPI_COMM_WORLD, &eigKSP);
  KSPSetOptionsPrefix(eigKSP,"eig_");
  KSPSetFromOptions(eigKSP);
  KSPSetOperators(eigKSP, damg[0]->J, damg[0]->B, SAME_NONZERO_PATTERN);
  Vec rhs;
  Vec sol;
  MatGetVecs(damg[0]->J, &sol, &rhs);
  computeDummyRHS(damg[0], rhs);
  KSPSetComputeEigenvalues(eigKSP, PETSC_TRUE);
  KSPSolve(eigKSP, rhs, sol);

  VecDestroy(rhs);
  VecDestroy(sol);
  KSPDestroy(eigKSP);

  VecDestroy(U);

  destroyLmat(LaplacianStencil);
  destroyGDmat(GradDivStencil);
  destroyPhimat(PhiMatStencil, numGpts);

  destroyHessContexts(damg);

  ot::DAMGDestroy(damg);

  ot::DAMG_Finalize();

  PetscFinalize();

}



