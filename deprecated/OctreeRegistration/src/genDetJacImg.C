
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#define DiffX(cArr, psi, eta, gamma) ((cArr[1]) + ((cArr[4])*(eta)) + \
    ((cArr[6])*(gamma)) + ((cArr[7])*(eta)*(gamma)))

#define DiffY(cArr, psi, eta, gamma) ((cArr[2]) + ((cArr[4])*(psi)) + \
    ((cArr[5])*(gamma)) + ((cArr[7])*(psi)*(gamma)))

#define DiffZ(cArr, psi, eta, gamma) ((cArr[3]) + ((cArr[5])*(eta)) + \
    ((cArr[6])*(psi)) + ((cArr[7])*(eta)*(psi)))


void loadSeqVector(std::vector<double> & v, char* fname);

void detJac(int Ne, int sliceNum, const std::vector<double>& u,
    std::vector<double>& img);

int main(int argc, char** argv) {

  if(argc < 5) {
    std::cout<<"Usage: exe dispFile Ne sliceNum(1-based) vtkFile "<<std::endl;
    exit(0);
  }

  std::vector<double> U;
  loadSeqVector(U, argv[1]);
  int Ne = atoi(argv[2]);
  int sliceNum = atoi(argv[3]);
  sliceNum--;

  std::vector<double> detJacImg;
  detJac(Ne, sliceNum, U, detJacImg);

  double h = 1.0/static_cast<double>(Ne);
  double z0 = static_cast<double>(sliceNum)*h;

  FILE* vfptr = fopen(argv[4], "w");
  fprintf(vfptr,"# vtk DataFile Version 3.0\n");
  fprintf(vfptr,"Volume File\n");
  fprintf(vfptr,"ASCII\n");
  fprintf(vfptr,"DATASET STRUCTURED_POINTS\n");
  fprintf(vfptr,"DIMENSIONS %d %d 2\n", Ne, Ne);
  fprintf(vfptr,"ORIGIN 0 0 %lf\n",z0);
  fprintf(vfptr,"SPACING %lf %lf %lf\n", h, h, (0.5*h));
  fprintf(vfptr,"POINT_DATA %d\n",(2*Ne*Ne));
  fprintf(vfptr,"SCALARS detJac float 1\n");
  fprintf(vfptr,"LOOKUP_TABLE default\n");

  for(int k = 0; k < 2; k++) {
    for(int i = 0; i < (Ne*Ne); i++) {
      fprintf(vfptr, "%f \n", (float)(detJacImg[i]));     
    }//end for i
  }//end for k

  fclose(vfptr);

}

void loadSeqVector(std::vector<double> & v, char* fname) {
  FILE* fptr = fopen(fname, "rb");

  int vlen;
  fread(&vlen, sizeof(int), 1, fptr);

  v.resize(vlen);

  fread((&(*(v.begin()))), sizeof(double), vlen, fptr);

  fclose(fptr);
}

void detJac(int Ne, int sliceNum, const std::vector<double>& u,
    std::vector<double>& img) {

  double h = 1.0/static_cast<double>(Ne);

  double Minv[][8] = { { 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250},
    { -0.1250, 0.1250, -0.1250, 0.1250, -0.1250, 0.1250, -0.1250, 0.1250},
    { -0.1250, -0.1250, 0.1250, 0.1250, -0.1250, -0.1250, 0.1250, 0.1250},
    { -0.1250, -0.1250, -0.1250, -0.1250, 0.1250, 0.1250, 0.1250, 0.1250},
    { 0.1250, -0.1250, -0.1250, 0.1250, 0.1250, -0.1250, -0.1250, 0.1250},
    { 0.1250, 0.1250, -0.1250, -0.1250, -0.1250, -0.1250, 0.1250, 0.1250},
    { 0.1250, -0.1250, 0.1250, -0.1250, -0.1250, 0.1250, -0.1250, 0.1250},
    { -0.1250, 0.1250, 0.1250, -0.1250, 0.1250, -0.1250, -0.1250, 0.1250} };

  img.resize(Ne*Ne);

  for(int j = 0; j < Ne; j++) {
    for(int i = 0; i < Ne; i++) {
      double coeffs[3][8];
      double vertices[3][8];
      int ilist[8];
      int jlist[8];
      int klist[8];
      for(int r = 0; r < 8; r++) {
        ilist[r] = i + (r%2);
        jlist[r] = j + ((r/2)%2);
        klist[r] = sliceNum + (r/4);
      }//end for r
      for(int r = 0; r < 8; r++) {
        double xyzPos[3];
        xyzPos[0] = static_cast<double>(ilist[r])*h;
        xyzPos[1] = static_cast<double>(jlist[r])*h;
        xyzPos[2] = static_cast<double>(klist[r])*h;
        int uIdx = (((klist[r]*(Ne + 1)) + jlist[r])*(Ne + 1)) + ilist[r];
        for(int d = 0; d < 3; d++) {
          vertices[d][r] = xyzPos[d] + u[(3*uIdx) + d];
        }//end for d
      }//end for r
      for(int d = 0; d < 3; d++) {
        for(int r = 0; r < 8; r++) {
          coeffs[d][r] = 0;
        }//end for r
      }//end for d
      for(int d = 0; d < 3; d++) {
        for(int r = 0; r < 8; r++) {
          for(int c = 0; c < 8; c++) {
            coeffs[d][r] += (Minv[r][c]*vertices[d][c]);
          }//end for c
        }//end for r
      }//end for d 
      double psi = 0;
      double eta = 0;
      double gamma = 0;
      double Jmat[3][3]; 
      for(int d = 0; d < 3; d++) {
        Jmat[d][0] = DiffX(coeffs[d], psi, eta, gamma); 
        Jmat[d][1] = DiffY(coeffs[d], psi, eta, gamma); 
        Jmat[d][2] = DiffZ(coeffs[d], psi, eta, gamma); 
      }//end for d 
      double detJacLocal = ( ( (Jmat[0][0]*Jmat[1][1]*Jmat[2][2]) +
            (Jmat[0][1]*Jmat[1][2]*Jmat[2][0]) + (Jmat[0][2]*Jmat[1][0]*Jmat[2][1]) ) -
          ( (Jmat[2][0]*Jmat[1][1]*Jmat[0][2]) + (Jmat[2][1]*Jmat[1][2]*Jmat[0][0]) +
            (Jmat[2][2]*Jmat[1][0]*Jmat[0][1]) ) )*8.0/(h*h*h);
      img[(j*Ne) + i] = detJacLocal;
    }//end for i
  }//end for j

}//end fn



