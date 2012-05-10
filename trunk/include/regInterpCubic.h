
#ifndef _REG_INTERP_CUBIC_H_
#define _REG_INTERP_CUBIC_H_

#include "mpi.h"
#include "oct/TreeNode.h"
#include "oda/oda.h"
#include <vector>

void computeNodalGradTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal,
    const std::vector<std::vector<double> > & gradTauLocal, int Ne, int padding, int numImages,
    Vec U, std::vector<double >& nodalGradTauAtU);

void computeNodalTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal,
    const std::vector<std::vector<double> > & gradTauLocal, int Ne, int padding, int numImages,
    Vec U, std::vector<double >& nodalTauAtU);

void computeSigVals(ot::DA* dao, const std::vector<std::vector<double> > & sigLocal, 
    const std::vector<std::vector<double> > & gradSigLocal,
    int Ne, int padding, int numImages, int numGpts, double* gPts,
    std::vector<double> & sigVals);

void computeGradTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal, 
    const std::vector<std::vector<double> > & gradTauLocal,
    int Ne, int padding, int numImages, Vec U, 
    double****** PhiMatStencil, int numGpts, double* gPts,
    std::vector<double> & gradTauAtU);

void computeTauAtU(ot::DA* dao, const std::vector<std::vector<double> > & tauLocal, 
    const std::vector<std::vector<double> > & gradTauLocal,
    int Ne, int padding, int numImages, Vec U, 
    double****** PhiMatStencil, int numGpts, double* gPts,
    std::vector<double> & tauAtU);

void computeRegularNodalTauAtU(int Ne, const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalTauAtU) ;
/**
  @param fLocal local image (first dimension identifies the block and the
  second dimension identifies the array index within the block)
  @param gLocal gradient of local image (first dimension identifies the block and the
  second dimension identifies the array index within the block)
  @param numImages number of features in image vector
  @param padding thickness (number of voxels) of buffer zone
  @param Ne number of elements in regular grid in each dimension
  @param xPosArr list of x-coordinates of points
  @param yPosArr list of y-coordinates of points
  @param zPosArr list of z-coordinates of points
  @param results list of image values at the specified points
  */

void evalFnAtAllPts(ot::DA* dao, const std::vector<std::vector<double> > &fLocal,
    const std::vector<std::vector<double> > &gLocal,
    int Ne, int padding, int numImages,
    const std::vector<double> & xPosArr, const std::vector<double> & yPosArr,
    const std::vector<double> & zPosArr, std::vector<double> & results);

void evalFnAtAllPts(const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    int Ne, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);


void evalGradFnAtAllPts(ot::DA* dao, const std::vector<std::vector<double> > &fLocal,
    const std::vector<std::vector<double> > &gLocal,
    int Ne, int padding, int numImages,
    const std::vector<double> & xPosArr, const std::vector<double> & yPosArr,
    const std::vector<double> & zPosArr, std::vector<double> & results);

void evalCubicGradFn(const std::vector<ot::TreeNode> & blocks,
    const std::vector<std::vector<double> > & fLocal,
    const std::vector<std::vector<double> > & gLocal,
    int Ne, int padding, int numImages,
    double xPos, double yPos, double zPos, std::vector<double>& results);

void evalCubicFn(const std::vector<ot::TreeNode> & blocks,
    const std::vector<std::vector<double> > & fLocal, 
    const std::vector<std::vector<double> > & gLocal,
    int Ne, int padding, int numImages,
    double xPos, double yPos, double zPos, std::vector<double>& results);

double evalCubicFnOld(const double* fArr, const double* gArr, 
    int Ne, double xPos, double yPos, double zPos);

void evalAll3Dcubic(double psi, double eta, double gamma, double phiArr[8][4]);

void evalAll3DcubicGrad(double psi, double eta, double gamma, double gradPhiArr[8][4][3]);

double eval1Dcubic(int nodeNum, int compNum, double psi);

double eval1DcubicGrad(int nodeNum, int compNum, double psi);

#endif 


