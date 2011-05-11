
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

  MPI_Comm commAll = MPI_COMM_WORLD;

  int rank, npesAll;
  MPI_Comm_rank(commAll, &rank);
  MPI_Comm_size(commAll, &npesAll);

  if(argc < 2) {
    if(!rank) {
      std::cout<<"Usage: exe fixedImg"<<std::endl;
    }
    PetscFinalize();
    exit(0);
  }

  unsigned int dim = 3;
  unsigned int dof = 3;  

  PetscInt maxDepth = 30;
  PetscOptionsGetInt(0, "-maxDepth", &maxDepth, 0);

  PetscReal threshold = 1.0;
  PetscOptionsGetReal(0, "-threshold", &threshold, 0);

  int Nfe;
  std::vector<double> sigImgFinest;
  std::vector<double> tauImgFinest;

  if(!rank) {
    struct dsr hdrSig;

    readImage(argv[1], &hdrSig, sigImgFinest);

    Nfe = hdrSig.dime.dim[1];

    assert(hdrSig.dime.dim[2] == Nfe);
    assert(hdrSig.dime.dim[3] == Nfe);
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

  assert(Nfe >= Nce);
  int numOptLevels = (binOp::fastLog2(Nfe/Nce)) + 1;

  std::vector<double>* sigImg = new std::vector<double>[numOptLevels];

  if(!rank) {
    //Loop finest to coarsest
    for(int lev = (numOptLevels - 1); lev >= 0; lev--) {
      int Ne = Nce*(1u << lev);
      if(Ne == Nfe) {
        sigImg[lev] = sigImgFinest;
        sigImgFinest.clear();
      } else {
        coarsenImage((2*Ne), sigImg[lev + 1], sigImg[lev]);
      }
    }//end lev
  }

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

  //Loop coarsest to finest
  for(int lev = 0; lev < 1; lev++) {
    int Ne = Nce*(1u << lev);

    if(rank < npesCurr[lev]) {
      DA da;
      DACreate3d(commCurr[lev], DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
          PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
          1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);

      PetscInt xs, ys, zs, nx, ny, nz;
      DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

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

      Vec sigN0;
      createSeqNodalImageVec(Ne, rank, npesCurr[lev], sigImg[lev], sigN0, commCurr[lev]);
      sigImg[lev].clear();

      Vec sigNatural;
      DACreateNaturalVector(da, &sigNatural);

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
          sendOff, recvSz, recvOff, commCurr[lev]);

      delete [] sendSz;
      delete [] recvSz;
      delete [] sendOff;
      delete [] recvOff;

      VecDestroy(sigN0);

      Vec sigGlobal;
      DACreateGlobalVector(da, &sigGlobal);
      DANaturalToGlobalBegin(da, sigNatural, INSERT_VALUES, sigGlobal);
      DANaturalToGlobalEnd(da, sigNatural, INSERT_VALUES, sigGlobal);

      VecDestroy(sigNatural);

      PetscScalar*** sigGlobalArr;
      DAVecGetArray(da, sigGlobal, &sigGlobalArr);

      std::vector<double> sigElemental;
      for(int k = zs; k < zs + nze; k++) {
        for(int j = ys; j < ys + nye; j++) {
          for(int i = xs; i < xs + nxe; i++) {
            double xPos = static_cast<double>(i)*h;
            double yPos = static_cast<double>(j)*h;
            double zPos = static_cast<double>(k)*h;
            double valSig = sigGlobalArr[k][j][i];
            sigElemental.push_back(valSig);
          }
        }
      }

      DAVecRestoreArray(da, sigGlobal, &sigGlobalArr);

      VecDestroy(sigGlobal);

      DADestroy(da);

      //Image to octree
      std::vector<ot::TreeNode> linSigOct;
      ot::regularGrid2Octree(sigElemental, Ne, nxe, nye, nze, xs, ys, zs,
          linSigOct, dim, maxDepth, threshold, commCurr[lev]);

      sigElemental.clear();

      char sigFname[256];
      sprintf(sigFname,"linSigOct_%d_%d_%d.ot", lev, rank, npesCurr[lev]);
      //ot::writeNodesToFile(sigFname, linSigOct);

    }//end if active

  }//end for lev

  delete [] commCurr;
  delete [] npesCurr;
  delete [] sigImg;

  PetscFinalize();

}



