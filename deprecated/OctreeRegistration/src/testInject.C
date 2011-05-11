
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

void zeroInt(int& v) {
  v = 0;
}

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, "optionsTest", 0);
  ot::RegisterEvents();
  ot::DAMG_Initialize(MPI_COMM_WORLD);

  int rank, npes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  ot::DAMG *damg;    
  unsigned int dim = 3;
  PetscInt maxDepth = 30;
  int nlevels = 1; 
  unsigned int dof = 3;  
  bool incCorner = 1;  
  bool compressLut = false;
  PetscReal mgLoadFac = 2.0;

  PetscInt nlevelsPetscInt = nlevels;
  PetscOptionsGetInt(0, "-nlevels", &nlevelsPetscInt, 0);
  nlevels = nlevelsPetscInt;

  PetscInt Ne;
  PetscOptionsGetInt(0, "-Ne", &Ne, 0);
  PetscOptionsGetInt(0, "-maxDepth", &maxDepth, 0);
  PetscOptionsGetReal(0, "-mgLoadFac", &mgLoadFac, 0);

  unsigned int regLev = binOp::fastLog2(Ne);

  std::vector<ot::TreeNode> balOct;
  ot::createRegularOctree(balOct, regLev, 3, maxDepth, MPI_COMM_WORLD);

  ot::DAMGCreateAndSetDA(MPI_COMM_WORLD, nlevels, NULL, &damg, 
      balOct, dof, mgLoadFac, compressLut, incCorner);
  balOct.clear();

  ot::PrintDAMG(damg);

  std::vector<int>* vecs = new std::vector<int>[nlevels];

  //Set the finest first
  ot::DA* daFinest = damg[nlevels - 1]->da;
  daFinest->createVector<int>(vecs[nlevels - 1], false, false, 1);

  int* arr;
  daFinest->vecGetBuffer<int>(vecs[nlevels - 1], arr, false, false, false, 1);

  double hRg = 1.0/static_cast<double>(Ne);
  if(daFinest->iAmActive()) {
    for(daFinest->init<ot::DA_FLAGS::ALL>();
        daFinest->curr() < daFinest->end<ot::DA_FLAGS::ALL>();
        daFinest->next<ot::DA_FLAGS::ALL>()) {
      Point pt = daFinest->getCurrentOffset();
      double x = static_cast<double>(pt.xint())/static_cast<double>(1u << maxDepth);
      double y = static_cast<double>(pt.yint())/static_cast<double>(1u << maxDepth);
      double z = static_cast<double>(pt.zint())/static_cast<double>(1u << maxDepth);
      int i = static_cast<int>(x/hRg);
      int j = static_cast<int>(y/hRg);
      int k = static_cast<int>(z/hRg); 
      unsigned int idx = daFinest->curr();
      unsigned char hnMask = daFinest->getHangingNodeIndex(idx);
      unsigned int indices[8];
      daFinest->getNodeIndices(indices);
      for(int v = 0; v < 8; v++) {
        if(!(hnMask & (1u << v))) {
          int iI = i + (v%2);
          int jI = j + ((v/2)%2);
          int kI = k + (v/4);
          arr[indices[v]] = (((kI*(Ne + 1)) + jI)*(Ne + 1)) + iI;
        }
      }//end for v
    }//end for ALL
  }

  daFinest->vecRestoreBuffer<int>(vecs[nlevels - 1], arr, false, false, false, 1);

  //Coarsen using injection
  for(int i = (nlevels - 1); i > 0; i--) {
    ot::DA* dac = damg[i - 1]->da;
    ot::DA* daf = NULL;
    std::vector<int>* cVec = vecs + (i - 1);
    std::vector<int>* fVec = NULL;

    if(damg[i]->da_aux) {
      daf = damg[i]->da_aux;
      fVec = new std::vector<int>; 
      par::scatterValues<int>(vecs[i], (*fVec), (daf->getNodeSize()), damg[0]->comm);
    } else {
      daf = damg[i]->da;
      fVec = vecs + i;
    } 

    ot::injectNodalVector<int>(dac, daf, 1, (*fVec), (*cVec), zeroInt);

    if(damg[i]->da_aux) {
      delete fVec;
    }
  }//end for i

  //Now Test
  for(int lev = (nlevels - 1); lev >= 0; lev--) {
    ot::DA* da = damg[lev]->da;
    int* buf;
    da->vecGetBuffer<int>(vecs[lev], buf, false, false, true, 1);

    da->ReadFromGhostsBegin<int>(buf, 1);
    da->ReadFromGhostsEnd<int>(buf);

    if(da->iAmActive()) {
      for(da->init<ot::DA_FLAGS::WRITABLE>();
          da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
          da->next<ot::DA_FLAGS::WRITABLE>()) {
        Point pt = da->getCurrentOffset();
        double x = static_cast<double>(pt.xint())/static_cast<double>(1u << maxDepth);
        double y = static_cast<double>(pt.yint())/static_cast<double>(1u << maxDepth);
        double z = static_cast<double>(pt.zint())/static_cast<double>(1u << maxDepth);
        int i = static_cast<int>(x/hRg);
        int j = static_cast<int>(y/hRg);
        int k = static_cast<int>(z/hRg); 
        unsigned int idx = da->curr();
        unsigned char hnMask = da->getHangingNodeIndex(idx);
        unsigned int indices[8];
        da->getNodeIndices(indices);
        for(int v = 0; v < 8; v++) {
          if(!(hnMask & (1u << v))) {
            int iI = i + ((v%2)*(1 << (nlevels - 1 - lev)));
            int jI = j + (((v/2)%2)*(1 << (nlevels - 1 - lev)));
            int kI = k + ((v/4)*(1 << (nlevels - 1 - lev)));
            if(buf[indices[v]] != ((((kI*(Ne + 1)) + jI)*(Ne + 1)) + iI) ) {
              std::cout<<"Proc "<<rank<<" failing for (i,j,k,v) = ("
                <<i<<", "<<j<<", "<<k<<", "<<v<<") "
                <<"Found: "<<buf[indices[v]]<<" Expected: "<<((((kI*(Ne + 1)) + jI)*(Ne + 1)) + iI)
                <<" index: "<<indices[v]<<" curr: "<<idx
                <<" myElemBeg: "<<da->getIdxElementBegin()
                <<" myElemEnd: "<<da->getIdxElementEnd()
                <<" myPostGhostBeg: "<<da->getIdxPostGhostBegin()
                <<std::endl;
            }
            assert(buf[indices[v]] == (((kI*(Ne + 1)) + jI)*(Ne + 1)) + iI);
          }
        }//end for v
      }//end for WRITABLE
    }

    da->vecRestoreBuffer<int>(vecs[lev], buf, false, false, true, 1);

  }//end for lev

  delete [] vecs;

  ot::DAMGDestroy(damg);

  ot::DAMG_Finalize();

  PetscFinalize();

}



