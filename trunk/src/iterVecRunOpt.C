
#include "mpi.h"
#include "regInterpCubic.h"
#include "registration.h"
#include <cmath>
#include <cstring>
#include "dendro.h"
#include <iostream>
#include "elasReg.h"
#include "elasVars.h"
#include <sys/time.h>



#define __PI__ 3.14159265

#define get_seconds()   (gettimeofday(&tp, &tzp), \
                        (double)tp.tv_sec + (double)tp.tv_usec / 1000000.0)



//Don't want time to be synchronized. Need to check load imbalance.
#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif


int main(int argc, char **argv){
  PetscInitialize(&argc,&argv,"options" ,0);
  if (argc <7){
    std::cout<<"Usage: exe targetScalarImg templateScalarImg displacement objFunc numIter DisplacedScalarImg"<<std::endl;
    PetscFinalize();
    exit(0);
  }
  else {
  MPI_Comm commAll = MPI_COMM_WORLD;

  int rank, npesAll;
  MPI_Comm_rank(commAll, &rank);
  MPI_Comm_size(commAll, &npesAll);

  char vecTarget[256],vecTemplate[256],vecDisplacedImg[256];
  sprintf(vecTarget,"vec%s",argv[1]);
  sprintf(vecTemplate,"vec%s",argv[2]);
  int numIters = atoi(argv[5]);
  int numImg=6;
  PetscOptionsGetInt(0,"-numImages", &numImg,0);
  int scalarImg=0;
  PetscOptionsGetInt(0,"-scalarImg", &scalarImg,0);
 
  if (!rank){
    if (scalarImg && numImg >1){
    fprintf(stdout,"write vector img for %s\n",argv[1]);
    writeVecImg(argv[1],vecTarget,numImg);
    fprintf(stdout,"write vector img for %s\n",argv[2]);
    writeVecImg(argv[2],vecTemplate,numImg);

    fprintf(stdout,"doing multi-step vector registration ...\n");
  }
   else {
    strcpy(vecTarget,argv[1]);
    strcpy(vecTemplate,argv[2]);
    }
  }
  MPI_Barrier(commAll);
 
  double vecRunTime=-get_seconds();  
  iterVecRunOptPar(vecTarget,vecTemplate,argv[3],argv[4],numIters);
  vecRunTime += get_seconds();

  MPI_Barrier(commAll); 
  if (!rank){
  printf("Total run time %f\n",vecRunTime);
  sprintf(vecDisplacedImg,"%s.%d",vecTemplate,numIters-1);
  if (scalarImg && numImg >1){
  fprintf(stdout,"convert vector img %s to scalar img %s\n",vecDisplacedImg,argv[6]);
  writeScalarImg(vecDisplacedImg,argv[6],numImg);
  }
  }
  PetscFinalize();
  }

}
