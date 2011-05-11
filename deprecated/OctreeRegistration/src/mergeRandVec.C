
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {

  if(argc < 3) {
    std::cout<<"Usage: exe npes fnamePrefix"<<std::endl;
    exit(0);
  }

  FILE* fp;
  char fname[256];
  int npes = atoi(argv[1]);

  int totalLen = 0;
  for(int i = 0; i < npes; i++) {
    sprintf(fname,"%s_%d_%d.dat", argv[2], i, npes);
    fp = fopen(fname, "rb");
    int localLen = 0;
    fread(&localLen, sizeof(int), 1, fp);  
    totalLen += localLen;    
    fclose(fp);
  }

  double* arr = new double[totalLen];

  totalLen = 0;
  for(int i = 0; i < npes; i++) {
    sprintf(fname,"%s_%d_%d.dat", argv[2], i, npes);
    fp = fopen(fname, "rb");
    int localLen = 0;
    fread(&localLen, sizeof(int), 1, fp);  
    fread(arr + totalLen, sizeof(double), localLen, fp);
    totalLen += localLen;      
    fclose(fp);
  }

  sprintf(fname,"%s_%d_%d.dat", argv[2], 0, 1);
  fp = fopen(fname, "wb");  
  fwrite(&totalLen, sizeof(int), 1, fp);
  fwrite(arr, sizeof(double), totalLen, fp);
  fclose(fp);

  delete [] arr;    
}


