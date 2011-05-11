
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

#define __PI__ 3.14159265

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);

  if(argc < 4) {
    std::cout<<"Usage: exe Ne scale outfile"<<std::endl;
    PetscFinalize();
    exit(0);
  }

  int rank, npes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int Ne = atoi(argv[1]);
  double scale = atof(argv[2]);

  //Convert from Morton ordering to DA ordering
  DA da;

  DACreate3d(MPI_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX, Ne + 1, Ne + 1, Ne + 1,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 3, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);

  Vec nvec;
  DACreateNaturalVector(da, &nvec);

  VecZeroEntries(nvec);

  double h = 1.0/static_cast<double>(Ne);

  PetscInt xs, ys, zs;
  PetscInt nx, ny, nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  for(int k = zs; k < (zs + nz); k++) {
    for(int j = ys; j < (ys + ny); j++) {
      for(int i = xs; i < (xs + nx); i++) {
        int row = (3*((((k*(Ne + 1)) + j)*(Ne + 1)) + i));
        double x = static_cast<double>(i)*h;
        double y = static_cast<double>(j)*h;
        double z = static_cast<double>(k)*h;
        double val = scale*0.001*sin(2.0*__PI__*x*64.0/5.0)*
          sin(2.0*__PI__*y*64.0/5.0)*sin(2.0*__PI__*z*64.0/5.0);
        VecSetValue(nvec, row, val, INSERT_VALUES);
      }
    }
  }

  VecAssemblyBegin(nvec);
  VecAssemblyEnd(nvec);

  Vec gvec;
  DACreateGlobalVector(da, &gvec);

  DANaturalToGlobalBegin(da, nvec, INSERT_VALUES, gvec);
  DANaturalToGlobalEnd(da, nvec, INSERT_VALUES, gvec);

  double maxDetJac, minDetJac;
  detJacMaxAndMin(da, gvec, &maxDetJac, &minDetJac);

  if(!rank) {
    std::cout<<" DetJac: max: "<<maxDetJac<<" min: "<<minDetJac<<std::endl;
  }

  char fname[256];
  sprintf(fname, "%s_%d_%d.dat", argv[3], rank, npes);
  saveVector(nvec, fname);

  VecDestroy(nvec);
  VecDestroy(gvec);

  DADestroy(da);

  PetscFinalize();

}



