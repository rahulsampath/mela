
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {

  if(argc < 5) {
    std::cout<<"Usage: exe Ne sliceNum (1-based) ptsFile coarsenFac "<<std::endl;
    exit(0);
  }

  int Ne = atoi(argv[1]);
  int sliceNum = atoi(argv[2]);
  FILE* outfile = fopen(argv[3],"w");
  int coarsenFac = atoi(argv[4]);

  sliceNum--;

  double h = 1.0/static_cast<double>(Ne);

  for(int j = 0; j < (Ne + 1); j+= coarsenFac) {
    for(int i = 0; i < (Ne + 1); i += coarsenFac) {
      float fx = (h*static_cast<double>(i));
      float fy = (h*static_cast<double>(j));
      float fz = (h*static_cast<double>(sliceNum));  
      fprintf(outfile,"%f %f %f \n",fx,fy,fz);
    }//end for i
  }//end for j

  fclose(outfile);

}


