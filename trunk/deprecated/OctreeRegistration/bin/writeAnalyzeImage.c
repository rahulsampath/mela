

#include "math.h"
#include "string.h"
#include "stdio.h"
#include "../include/dbh.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  char* fnamePrefix = mxArrayToString(prhs[0]);
  int Nx = (int)(mxGetScalar(prhs[1]));
  int Ny = (int)(mxGetScalar(prhs[2]));
  int Nz = (int)(mxGetScalar(prhs[3]));
  double hx = mxGetScalar(prhs[4]);
  double hy = mxGetScalar(prhs[5]);
  double hz = mxGetScalar(prhs[6]);
  double* img = mxGetPr(prhs[7]);
  char fname[256];
  FILE* fp;
  struct dsr hdr;
  int maxVal, minVal;
  int totalPixels;
  int i;

  memset(&hdr, 0, sizeof(struct dsr));
  for(i = 0; i < 8; i++) {
    hdr.dime.pixdim[i] = 0.0;
  }

  hdr.dime.vox_offset = 0.0;
  hdr.dime.funused1 = 0.0;
  hdr.dime.funused2 = 0.0;
  hdr.dime.funused3 = 0.0;
  hdr.dime.cal_max = 0.0;
  hdr.dime.cal_min = 0.0;

  /* DOUBLE */
  hdr.dime.datatype = 64;
  hdr.dime.bitpix = 64;

  sprintf(fname, "%s.hdr", fnamePrefix);
  fp = fopen(fname, "wb");

  hdr.dime.dim[0] = 4; /* all Analyze images are taken as 4 dimensional */
  hdr.hk.regular = 'r';
  hdr.hk.sizeof_hdr = sizeof(struct dsr);

  hdr.dime.dim[1] = Nx; /* slice width in pixels */
  hdr.dime.dim[2] = Ny; /* slice height in pixels */
  hdr.dime.dim[3] = Nz; /* volume depth in slices */
  hdr.dime.dim[4] = 1; /* number of volumes per file */

  maxVal, minVal;
  maxVal = img[0];
  minVal = img[0];
  totalPixels = Nx*Ny*Nz;
  for(i = 0; i < totalPixels; i++) {
    if(maxVal < img[i]) {
      maxVal = img[i];
    }
    if(minVal > img[i]) {
      minVal = img[i];
    }
  }

  hdr.dime.glmax = maxVal; /* maximum voxel value */
  hdr.dime.glmin = minVal; /* minimum voxel value */

  /* Set the voxel dimension fields:
     A value of 0.0 for these fields implies that the value is unknown.
     Change these values to what is appropriate for your data
     or pass additional command line arguments */

  hdr.dime.pixdim[1] = hx; /* voxel x dimension */
  hdr.dime.pixdim[2] = hy; /* voxel y dimension */
  hdr.dime.pixdim[3] = hz; /* pixel z dimension, slice thickness */

  /* Assume zero offset in .img file, byte at which pixel
     data starts in the image file */

  hdr.dime.vox_offset = 0.0;

  /* Planar Orientation; */
  /* Movie flag OFF: 0 = transverse, 1 = coronal, 2 = sagittal
     Movie flag ON: 3 = transverse, 4 = coronal, 5 = sagittal */

  hdr.hist.orient = 0;

  /* up to 3 characters for the voxels units label; i.e. mm., um., cm. */ 

  strcpy(hdr.dime.vox_units," ");

  /* up to 7 characters for the calibration units label; i.e. HU */

  strcpy(hdr.dime.cal_units," ");

  /* Calibration maximum and minimum values;
     values of 0.0 for both fields imply that no
     calibration max and min values are used */

  hdr.dime.cal_max = 0.0;
  hdr.dime.cal_min = 0.0;

  fwrite(&hdr, sizeof(struct dsr), 1, fp);

  fclose(fp);

  sprintf(fname, "%s.img", fnamePrefix);
  fp = fopen(fname, "wb");

  fwrite(img, sizeof(double), totalPixels, fp);
  fclose(fp);

}


