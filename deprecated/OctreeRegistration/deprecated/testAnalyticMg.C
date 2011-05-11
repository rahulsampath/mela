
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
  PetscReal alpha = 1.0;
  PetscReal threshold = 1.0;
  PetscReal analyticImgMax = 255.0;
  bool incCorner = 1;  
  bool compressLut = false;
  PetscReal mgLoadFac = 2.0;

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
  PetscOptionsGetReal(0, "-alpha", &alpha, 0);
  PetscOptionsGetReal(0, "-mgLoadFac", &mgLoadFac, 0);
  PetscOptionsGetReal(0, "-threshold", &threshold, 0);
  PetscOptionsGetReal(0, "-analyticImgMax", &analyticImgMax, 0);

  DA da1dof;
  DA da3dof;

  DACreate3d(MPI_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da1dof);

  DACreate3d(MPI_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da3dof);

  Vec tauGlobal, gradTauGlobal;
  Vec tauNatural, gradTauNatural;
  Vec tauNaturalAll, gradTauNaturalAll;
  VecScatter toLocalAll1dof;
  VecScatter toLocalAll3dof;

  DACreateGlobalVector(da1dof, &tauGlobal);
  DACreateGlobalVector(da3dof, &gradTauGlobal);

  DACreateNaturalVector(da1dof, &tauNatural);
  DACreateNaturalVector(da3dof, &gradTauNatural);

  VecScatterCreateToAll(tauNatural, &toLocalAll1dof, &tauNaturalAll);
  VecScatterCreateToAll(gradTauNatural, &toLocalAll3dof, &gradTauNaturalAll);

  PetscScalar*** tauGlobalArr;
  DAVecGetArray(da1dof, tauGlobal, &tauGlobalArr);
  double h = 1.0/static_cast<double>(Ne);

  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  DAGetCorners(da1dof, &xs, &ys, &zs, &nx, &ny, &nz);
  for(int k = zs; k < zs + nz; k++) {
    for(int j = ys; j < ys + ny; j++) {
      for(int i = xs; i < xs + nx; i++) {
        double xPos = static_cast<double>(i)*h;
        double yPos = static_cast<double>(j)*h;
        double zPos = static_cast<double>(k)*h;
        tauGlobalArr[k][j][i] = analyticImgMax*sin(2.0*__PI__*xPos)*sin(2.0*__PI__*yPos)*sin(2.0*__PI__*zPos);
      }
    }
  }

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
        double val = analyticImgMax*sin(2.0*__PI__*xPos)*sin(2.0*__PI__*yPos)*sin(2.0*__PI__*zPos);
        tauElemental.push_back(val);
      }
    }
  }

  DAVecRestoreArray(da1dof, tauGlobal, &tauGlobalArr);

  computeFDgradient(da1dof, da3dof, tauGlobal, gradTauGlobal);

  DAGlobalToNaturalBegin(da1dof, tauGlobal, INSERT_VALUES, tauNatural);
  DAGlobalToNaturalBegin(da3dof, gradTauGlobal, INSERT_VALUES, gradTauNatural);

  DAGlobalToNaturalEnd(da1dof, tauGlobal, INSERT_VALUES, tauNatural);
  DAGlobalToNaturalEnd(da3dof, gradTauGlobal, INSERT_VALUES, gradTauNatural);

  VecScatterBegin(toLocalAll1dof, tauNatural, tauNaturalAll, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterBegin(toLocalAll3dof, gradTauNatural, gradTauNaturalAll, INSERT_VALUES, SCATTER_FORWARD );

  VecScatterEnd(toLocalAll1dof, tauNatural, tauNaturalAll, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(toLocalAll3dof, gradTauNatural, gradTauNaturalAll, INSERT_VALUES, SCATTER_FORWARD);

  PetscReal tauGlobalNorm, tauNaturalNorm, tauNaturalAllNorm;
  PetscReal gradTauGlobalNorm, gradTauNaturalNorm, gradTauNaturalAllNorm;

  VecNorm(tauGlobal, NORM_2, &tauGlobalNorm);
  VecNorm(gradTauGlobal, NORM_2, &gradTauGlobalNorm);

  VecNorm(tauNatural, NORM_2, &tauNaturalNorm);
  VecNorm(gradTauNatural, NORM_2, &gradTauNaturalNorm);

  VecNorm(tauNaturalAll, NORM_2, &tauNaturalAllNorm);
  VecNorm(gradTauNaturalAll, NORM_2, &gradTauNaturalAllNorm);

  std::cout<<"rank = "<<rank
    <<" tauGlobalNorm: "<<tauGlobalNorm
    <<" gradTauGlobalNorm: "<<gradTauGlobalNorm
    <<" tauNaturalNorm: "<<tauNaturalNorm
    <<" gradTauNaturalNorm: "<<gradTauNaturalNorm
    <<" tauNaturalAllNorm: "<<tauNaturalAllNorm
    <<" gradTauNaturalAllNorm: "<<gradTauNaturalAllNorm
    <<std::endl; 

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

  std::cout<<"rank = "<<rank<<" tau[z = 13][y = 15][x = 17] = "<<
    tau[0][(((13*(Ne + 1)) + 15)*(Ne + 1)) + 17]<<std::endl;
  std::cout<<"rank = "<<rank<<" gradTau1[z = 13][y = 15][x = 17] = "<<
    gradTau[0][3*((((13*(Ne + 1)) + 15)*(Ne + 1)) + 17)]<<std::endl;
  double xPos = static_cast<double>(17)*h;
  double yPos = static_cast<double>(15)*h;
  double zPos = static_cast<double>(13)*h;
  double trueVal = analyticImgMax*sin(2.0*__PI__*xPos)*sin(2.0*__PI__*yPos)*sin(2.0*__PI__*zPos);
  std::cout<<"trueVal = "<<trueVal<<std::endl;

  VecRestoreArray(tauNaturalAll, &tauNaturalAllArr);
  VecRestoreArray(gradTauNaturalAll, &gradTauNaturalAllArr);

  VecScatterDestroy(toLocalAll1dof);
  VecScatterDestroy(toLocalAll3dof);

  VecDestroy(tauNatural);
  VecDestroy(gradTauNatural);

  VecDestroy(tauNaturalAll);
  VecDestroy(gradTauNaturalAll);

  VecDestroy(tauGlobal);
  VecDestroy(gradTauGlobal);

  DADestroy(da1dof);
  DADestroy(da3dof);

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

  ot::PrintDAMG(damg);

  Vec U;
  damg[nlevels - 1]->da->createVector(U, false, false, 3);
  PetscRandom rctx;  
  PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
  PetscRandomSetType(rctx, PETSCRAND48);
  PetscInt randomSeed = 1652;
  PetscRandomSetSeed(rctx, randomSeed);
  PetscRandomSeed(rctx);
  PetscRandomSetFromOptions(rctx);
  VecSetRandom(U, rctx);

  PetscReal uNorm;
  VecNorm(U, NORM_2, &uNorm);

  if(!rank) {
    std::cout<<"U-Norm: "<<uNorm<<std::endl;
  }

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

  ot::DAMGSolve(damg);

  VecDestroy(U);
  PetscRandomDestroy(rctx);

  destroyLmat(LaplacianStencil);
  destroyGDmat(GradDivStencil);
  destroyPhimat(PhiMatStencil, numGpts);

  destroyHessContexts(damg);

  ot::DAMGDestroy(damg);

  ot::DAMG_Finalize();

  PetscFinalize();

}



