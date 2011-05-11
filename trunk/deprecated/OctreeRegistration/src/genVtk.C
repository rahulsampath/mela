
#include "mpi.h"
#include "regInterpCubic.h"
#include "registration.h"
#include <cmath>
#include "externVars.h"
#include "dendro.h"
#include <iostream>

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

  PetscInitialize(&argc, &argv, 0, 0);

  if(argc < 7) {
    std::cout<<"Usage: exe dispFile dispScale Ne sliceNum (1-based) vtkFile coarsenFac "<<std::endl;
    PetscFinalize();
    exit(0);
  }

  Vec U;
  loadSeqVector(U, argv[1]);
  double dispScale = atof(argv[2]);
  int Ne = atoi(argv[3]);
  int sliceNum = atoi(argv[4]);
  FILE* outfile = fopen(argv[5],"w");
  int coarsenFac = atoi(argv[6]);

  int numPtsPerDim = 0;
  for(int i = 0; i < (Ne + 1); i += coarsenFac) {
    numPtsPerDim++;
  }//end for i

  VecScale(U, dispScale);

  sliceNum--;

  fprintf(outfile,"# vtk DataFile Version 3.0\n");
  fprintf(outfile,"Surface Mesh file\n");
  fprintf(outfile,"ASCII\n");
  fprintf(outfile,"DATASET UNSTRUCTURED_GRID\n");
  fprintf(outfile,"POINTS %d float\n", (numPtsPerDim*numPtsPerDim));

  double h = 1.0/static_cast<double>(Ne);

  PetscScalar* uArr;
  VecGetArray(U, &uArr);

  for(int j = 0; j < (Ne + 1); j+= coarsenFac) {
    for(int i = 0; i < (Ne + 1); i += coarsenFac) {
      int idx = (((sliceNum*(Ne + 1)) + j)*(Ne + 1)) + i;
      float fx = (h*static_cast<double>(i)) + uArr[(3*idx)];
      float fy = (h*static_cast<double>(j)) + uArr[(3*idx) + 1];
      float fz = (h*static_cast<double>(sliceNum)) + uArr[(3*idx) + 2];  
      fprintf(outfile,"%f %f %f \n",fx,fy,fz);
    }//end for i
  }//end for j

  VecRestoreArray(U, &uArr);
  VecDestroy(U);

  fprintf(outfile,"\nCELLS %d %d\n", ((numPtsPerDim-1)*(numPtsPerDim-1)), (5*(numPtsPerDim - 1)*(numPtsPerDim - 1)));

  for(int j = 0; j < (numPtsPerDim - 1); j++) {
    for(int i = 0; i < (numPtsPerDim - 1); i++) {
      fprintf(outfile,"4 ");		
      {
        int xi = i;
        int yi = j;
        int idx = (yi*numPtsPerDim) + xi;
        fprintf(outfile,"%d ", idx);
      }
      {
        int xi = i + 1;
        int yi = j;
        int idx = (yi*numPtsPerDim) + xi;
        fprintf(outfile,"%d ", idx);
      }
      {
        int xi = i + 1;
        int yi = j + 1;
        int idx = (yi*numPtsPerDim) + xi;
        fprintf(outfile,"%d ", idx);
      }
      {
        int xi = i;
        int yi = j + 1;
        int idx = (yi*numPtsPerDim) + xi;
        fprintf(outfile,"%d ", idx);
      }
      fprintf(outfile,"\n");
    } //end for i
  }//end for j

  fprintf(outfile,"\nCELL_TYPES %d\n",((numPtsPerDim - 1)*(numPtsPerDim - 1)));

  for(int i = 0; i < ((numPtsPerDim - 1)*(numPtsPerDim - 1)); i++) {
    fprintf(outfile,"9 \n");
  }//end for i

  fprintf(outfile,"\nCELL_DATA %d\n", ((numPtsPerDim - 1)*(numPtsPerDim - 1)));
  fprintf(outfile,"SCALARS scalars unsigned_int\n");
  fprintf(outfile,"LOOKUP_TABLE default\n");

  for(int i = 0; i < ((numPtsPerDim - 1)*(numPtsPerDim - 1)); i++) {
    fprintf(outfile,"1 \n");     
  }//end for i

  fclose(outfile);

  PetscFinalize();
}


