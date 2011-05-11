
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <iostream>

void loadSeqVector(std::vector<double> & v, char* fname);

int main(int argc, char** argv) {

  if(argc < 6) {
    std::cout<<"Usage: exe dispFile Ne coarsenFac inPtsFile outPtsFile "<<std::endl;
    exit(0);
  }

  double shCoeffs[][8] = { {0.1250,   -0.1250,   -0.1250,   -0.1250,    0.1250,    0.1250,    0.1250,   -0.1250},
    { 0.1250,    0.1250,   -0.1250,   -0.1250,   -0.1250,    0.1250,   -0.1250,    0.1250},
    { 0.1250,   -0.1250,    0.1250,   -0.1250,   -0.1250,   -0.1250,    0.1250,    0.1250},
    { 0.1250,    0.1250,    0.1250,   -0.1250,    0.1250,   -0.1250,   -0.1250,   -0.1250},
    { 0.1250,   -0.1250,   -0.1250,    0.1250,    0.1250,   -0.1250,   -0.1250,    0.1250},
    { 0.1250,    0.1250,   -0.1250,    0.1250,   -0.1250,   -0.1250,    0.1250,   -0.1250},
    { 0.1250,   -0.1250,    0.1250,    0.1250,   -0.1250,    0.1250,   -0.1250,   -0.1250},
    { 0.1250,   0.1250,    0.1250,    0.1250,    0.1250,    0.1250,    0.1250,    0.1250} };

  std::vector<double> U;
  loadSeqVector(U, argv[1]);
  int Ne = atoi(argv[2]);
  int coarsenFac = atoi(argv[3]);
  FILE* inPtsFile = fopen(argv[4],"r");
  FILE* outPtsFile = fopen(argv[5],"w");

  int numPtsPerDim = 0;
  for(int i = 0; i < (Ne + 1); i += coarsenFac) {
    numPtsPerDim++;
  }//end for i

  double h = 1.0/static_cast<double>(Ne);

  for(int j = 0; j < numPtsPerDim; j++) {
    for(int i = 0; i < numPtsPerDim; i++) {
      double xi, yi, zi, xo, yo, zo, ux, uy, uz;
      fscanf(inPtsFile, "%lf", &xi);
      fscanf(inPtsFile, "%lf", &yi);
      fscanf(inPtsFile, "%lf", &zi);

      ux = 0;
      uy = 0;
      uz = 0;

      if( (xi >= 0.0) && (xi < 1.0) &&
          (yi >= 0.0) && (yi < 1.0) &&
          (zi >= 0.0) && (zi < 1.0) ) {
        int xe = static_cast<int>(xi/h);
        int ye = static_cast<int>(yi/h);
        int ze = static_cast<int>(zi/h);

        double psi = -1.0 + ((xi - (static_cast<double>(xe)*h))*2.0/h);
        double eta = -1.0 + ((yi - (static_cast<double>(ye)*h))*2.0/h);
        double gamma = -1.0 + ((zi - (static_cast<double>(ze)*h))*2.0/h);

        for(int l = 0; l < 8; l++) {
          int xn = xe + (l%2);
          int yn = ye + ((l/2)%2);
          int zn = ze + (l/4);
          int idx = (((zn*(Ne + 1)) + yn)*(Ne + 1)) + xn;

          double shVal = shCoeffs[l][0] + 
            (shCoeffs[l][1]*psi) + 
            (shCoeffs[l][2]*eta) + 
            (shCoeffs[l][3]*gamma) + 
            (shCoeffs[l][4]*psi*eta) + 
            (shCoeffs[l][5]*eta*gamma) + 
            (shCoeffs[l][6]*gamma*psi) + 
            (shCoeffs[l][7]*psi*eta*gamma);

          ux += (U[(3*idx)]*shVal);
          uy += (U[(3*idx) + 1]*shVal);
          uz += (U[(3*idx) + 2]*shVal);
        }//end for l
      }

      xo = xi + ux;
      yo = yi + uy;
      zo = zi + uz;

      fprintf(outPtsFile, "%lf %lf %lf \n", xo, yo, zo);
    }//end for i
  }//end for j

  fclose(inPtsFile);
  fclose(outPtsFile);

}

void loadSeqVector(std::vector<double> & v, char* fname) {
  FILE* fptr = fopen(fname, "rb");

  int vlen;
  fread(&vlen, sizeof(int), 1, fptr);

  v.resize(vlen);

  fread((&(*(v.begin()))), sizeof(double), vlen, fptr);

  fclose(fptr);
}


