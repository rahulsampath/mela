
#include "mpi.h"
#include "sys.h"
#include "omg.h"
#include <cstdlib>
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

double gaussian(double mean, double std_deviation);

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);

  ot::RegisterEvents();
  PetscLogEventRegister(&elasMultEvent, "ElasMatMult", PETSC_VIEWER_COOKIE);

  ot::DA_Initialize(MPI_COMM_WORLD);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  unsigned int dim = 3;
  PetscInt maxDepth = 30;
  unsigned int dof = 3;  
  double mu = 1.0;
  double lambda = 4.0;
  double alpha = 1.0;
  int numLoops = 50;

  unsigned int regLev = 4;
  unsigned int local_num_pts = 5000;
  bool usePts = false;

  if(argc > 1) {
    usePts = atoi(argv[1]);
  }

  if(argc > 2) {
    if(usePts) {
      local_num_pts = atoi(argv[2]);
    } else {
      regLev = atoi(argv[2]);
    }
  }

  if(!rank) {
    if(usePts) {
      std::cout<<"Using "<<local_num_pts<<" points."<<std::endl;
    } else {
      std::cout<<"Using regular lev: "<<regLev<<std::endl;
    }
  }

  std::vector<ot::TreeNode> balOct;
  if(usePts) {

    srand48(0x12345678 + 76543*rank);

    std::vector<double> pts;
    pts.resize(3*local_num_pts);
    for(int i = 0; i < (3*local_num_pts); i++) {
      pts[i]= gaussian(0.5, 0.16);
    }

    unsigned int ptsLen = pts.size();

    std::vector<ot::TreeNode> tmpNodes;
    for(int i = 0; i < ptsLen; i += 3) {
      if( (pts[i] > 0.0) &&
          (pts[i+1] > 0.0)  
          && (pts[i+2] > 0.0) &&
          ( ((unsigned int)(pts[i]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
          ( ((unsigned int)(pts[i+1]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
          ( ((unsigned int)(pts[i+2]*((double)(1u << maxDepth)))) < (1u << maxDepth)) ) {
        tmpNodes.push_back( ot::TreeNode((unsigned int)(pts[i]*(double)(1u << maxDepth)),
              (unsigned int)(pts[i+1]*(double)(1u << maxDepth)),
              (unsigned int)(pts[i+2]*(double)(1u << maxDepth)),
              maxDepth,dim,maxDepth) );
      }
    }
    pts.clear();

    par::removeDuplicates<ot::TreeNode>(tmpNodes, false, MPI_COMM_WORLD);	

    std::vector<ot::TreeNode> linOct  = tmpNodes;

    tmpNodes.clear();

    par::partitionW<ot::TreeNode>(linOct, NULL, MPI_COMM_WORLD);

    pts.resize(3*(linOct.size()));
    ptsLen = (3*(linOct.size()));
    for(int i = 0; i < linOct.size(); i++) {
      pts[3*i] = (((double)(linOct[i].getX())) + 0.5)/((double)(1u << maxDepth));
      pts[(3*i)+1] = (((double)(linOct[i].getY())) +0.5)/((double)(1u << maxDepth));
      pts[(3*i)+2] = (((double)(linOct[i].getZ())) +0.5)/((double)(1u << maxDepth));
    }//end for i

    linOct.clear();

    unsigned int maxNumPts = 1;
    double gSize[3];
    gSize[0] = 1.0;
    gSize[1] = 1.0;
    gSize[2] = 1.0;

    ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, MPI_COMM_WORLD);
    pts.clear();

    ot::balanceOctree (linOct, balOct, dim, maxDepth, true, MPI_COMM_WORLD, NULL, NULL);
    linOct.clear();

  } else {
    ot::createRegularOctree(balOct, regLev, dim, maxDepth, MPI_COMM_WORLD);
  }

  MPI_Comm activeComm;
  par::splitComm2way(balOct.empty(), &activeComm, MPI_COMM_WORLD);

  ot::DA* da = new ot::DA(balOct, MPI_COMM_WORLD, activeComm, false, NULL, NULL);

  balOct.clear();

  Vec in;
  Vec out;

  da->createVector(in, false, false, dof);

  VecDuplicate(in, &out);

  PetscRandom rctx;  
  PetscRandomCreate(MPI_COMM_WORLD, &rctx);
  PetscRandomSetType(rctx, PETSCRAND48);
  PetscInt randomSeed = 1652;
  PetscRandomSetSeed(rctx, randomSeed);
  PetscRandomSeed(rctx);
  PetscRandomSetFromOptions(rctx);
  VecSetRandom(in, rctx);
  PetscRandomDestroy(rctx);

  double**** LaplacianStencil; 
  double**** GradDivStencil;

  createLmat(LaplacianStencil);
  createGDmat(GradDivStencil);

  std::vector<unsigned char> bdyFlags;
  unsigned char* bdyArr = NULL;

  //This will create a nodal, non-ghosted, 1 dof array
  assignBoundaryFlags(da, bdyFlags);
  da->vecGetBuffer<unsigned char>(bdyFlags, bdyArr, false, false, true, 1);

  for(int i = 0; i < numLoops; i++) {
    elasMatVec(da, bdyArr, LaplacianStencil, GradDivStencil, mu, lambda, in, out);
  }

  da->vecRestoreBuffer<unsigned char>(bdyFlags, bdyArr, false, false, true, 1);
  bdyFlags.clear();

  VecDestroy(in);
  VecDestroy(out);

  destroyLmat(LaplacianStencil);
  destroyGDmat(GradDivStencil);

  delete da;

  ot::DA_Finalize();

  PetscFinalize();

}

double gaussian(double mean, double std_deviation) {
  static double t1 = 0, t2=0;
  double x1, x2, x3, r;

  using namespace std;

  // reuse previous calculations
  if(t1) {
    const double tmp = t1;
    t1 = 0;
    return mean + std_deviation * tmp;
  }
  if(t2) {
    const double tmp = t2;
    t2 = 0;
    return mean + std_deviation * tmp;
  }

  // pick randomly a point inside the unit disk
  do {
    x1 = 2 * drand48() - 1;
    x2 = 2 * drand48() - 1;
    x3 = 2 * drand48() - 1;
    r = x1 * x1 + x2 * x2 + x3*x3;
  } while(r >= 1);

  // Box-Muller transform
  r = sqrt(-2.0 * log(r) / r);

  // save for next call
  t1 = (r * x2);
  t2 = (r * x3);

  return mean + (std_deviation * r * x1);
}//end gaussian




