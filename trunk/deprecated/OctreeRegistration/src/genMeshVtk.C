
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {

  if(argc < 5) {
    std::cout<<"Usage: exe Ne coarsenFac ptsFile vtkFile "<<std::endl;
    exit(0);
  }

  int Ne = atoi(argv[1]);
  int coarsenFac = atoi(argv[2]);
  FILE* infile = fopen(argv[3],"r");
  FILE* outfile = fopen(argv[4],"w");

  int numPtsPerDim = 0;
  for(int i = 0; i < (Ne + 1); i += coarsenFac) {
    numPtsPerDim++;
  }//end for i

  fprintf(outfile, "# vtk DataFile Version 3.0\n");
  fprintf(outfile, "Surface Mesh file\n");
  fprintf(outfile, "ASCII\n");
  fprintf(outfile, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(outfile, "POINTS %d float\n", (numPtsPerDim*numPtsPerDim));

  for(int j = 0; j < numPtsPerDim; j++) {
    for(int i = 0; i < numPtsPerDim; i++) {
      double x, y, z;
      fscanf(infile, "%lf", &x);
      fscanf(infile, "%lf", &y);
      fscanf(infile, "%lf", &z);
      fprintf(outfile, "%f %f %f \n", (float)x, (float)y, (float)z);
    }//end for i
  }//end for j

  fclose(infile);

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

}


