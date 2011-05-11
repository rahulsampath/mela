
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {

  FILE* fp;
  char fname[256];
  int npes = atoi(argv[1]);

  for(int i = 1; i < npes; i++) {
    int vecLen = 0;
    sprintf(fname,"%s_%d_%d.dat", argv[2], i, npes);
    fp = fopen(fname, "wb");
    fwrite(&vecLen, sizeof(int), 1, fp);
    fclose(fp);
  }

}

