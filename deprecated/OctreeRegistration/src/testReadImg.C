
#include "mpi.h"
#include "registration.h"
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

  struct dsr hdr;

  std::vector<double> img;

  readImage(argv[1], &hdr, img);

  int dimX = hdr.dime.dim[1];
  int dimY = hdr.dime.dim[2];
  int dimZ = hdr.dime.dim[3];

  std::cout<<" img.size = "<<img.size()<<std::endl;
  std::cout<<" dimX = "<<dimX<<" dimY = "<<dimY<<" dimZ = "<<dimZ<<std::endl;
  std::cout<<" bitPix = "<<hdr.dime.bitpix<<std::endl;
  std::cout<<" img[x = 2][ y = 3][z = 4] = "<<img[(((3*dimY) + 2)*dimX) + 1]<<std::endl;

}

