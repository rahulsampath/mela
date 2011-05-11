/*#include "sys.h"
//#include "omg.h"
#include "registration.h"
#include "regInterpCubic.h"
#include "dendro.h"
#include <cstring>

#include "elasReg.h"

#include "elasVars.h"
*/
#include "mpi.h"
#include "regInterpCubic.h"
#include "registration.h"
#include <cmath>
#include "dendro.h"
#include <iostream>
#include "elasReg.h"
#include "elasVars.h"



#define __PI__ 3.14159265

//Don't want time to be synchronized. Need to check load imbalance.
#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

static char help[] = "Single solve of optimization code\n";


int main(int argc, char **argv){
  PetscInitialize(&argc,&argv,"options" ,0);
  if (argc <6){
    std::cout<<"Usage: exe targetImg templateImg displacement objFunc numIter"<<std::endl;
    PetscFinalize();
    exit(0);
  }
  else  iterRunOpt(argv[1],argv[2],argv[3],argv[4],atoi(argv[5]));
  PetscFinalize();
 

}
