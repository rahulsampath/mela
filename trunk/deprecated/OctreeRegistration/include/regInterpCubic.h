
#ifndef _REG_INTERP_CUBIC_H_
#define _REG_INTERP_CUBIC_H_

#include "oda.h"
#include "petscda.h"

/**
  @brief interpolates gradient of moving image at gauss points
  @param dao octree mesh
  @param patchwidth thickness of image buffer layer
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param tauLocal local portion of moving image
  @param gradTauLocal local portion of gradient of moving image
  @param U displacement
  @param PhiMatStencil shape function evaluated at gauss points
  @param numGpts number of gauss points
  @param gPts values for gauss points
  @param gradTauAtU result
  */
void computeNewGradTauAtU(ot::DA* dao, double patchWidth, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU); 

/**
  @brief interpolates moving image at gauss points
  @param dao octree mesh
  @param patchwidth thickness of image buffer layer
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param tauLocal local portion of moving image
  @param gradTauLocal local portion of gradient of moving image
  @param U displacement
  @param PhiMatStencil shape function evaluated at gauss points
  @param numGpts number of gauss points
  @param gPts values for gauss points
  @param tauAtU result
  */
void computeNewTauAtU(ot::DA* dao, double patchWidth, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU);

/**
  @brief interpolates fixed image at gauss points
  @param dao octree mesh
  @param patchwidth thickness of image buffer layer
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param sigLocal local portion of fixed image
  @param gradSigLocal local portion of gradient of fixed image
  @param numGpts number of gauss points
  @param gPts values for gauss points
  @param sigVals result
  */
void computeNewSigVals(ot::DA* dao, double patchWidth, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& sigLocal,
    const std::vector<std::vector<double> >& gradSigLocal, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals);

/**
  @brief interpolates gradient of moving image at mesh points
  @param dao octree mesh
  @param patchwidth thickness of image buffer layer
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param tauLocal local portion of moving image
  @param gradTauLocal local portion of gradient of moving image
  @param U displacement
  @param nodalGradTauAtU result
  */
void computeNewNodalGradTauAtU(ot::DA* dao, double patchWidth,
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal,
    Vec U, std::vector<std::vector<double> >& nodalGradTauAtU);

/**
  @brief interpolates moving image at mesh points
  @param dao octree mesh
  @param patchwidth thickness of image buffer layer
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param tauLocal local portion of moving image
  @param gradTauLocal local portion of gradient of moving image
  @param U displacement
  @param nodalTauAtU result
  */
void computeNewNodalTauAtU(ot::DA* dao, double patchWidth, 
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal,
    Vec U, std::vector<std::vector<double> >& nodalTauAtU);

/**
  @brief interpolates image at given points
  @param dao octree mesh
  @param xPosArr x coordinates of points
  @param yPosArr y coordinates of points
  @param zPosArr z coordinates of points
  @param outOfBndsList indices of points outside local domain
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param F image
  @param gradF gradient of image
  @param results result
  */
void newEvalFnAtAllPts(ot::DA* dao, const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    const std::vector<unsigned int>& outOfBndsList,
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& F, const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

/**
  @brief interpolates gradient of image at given points
  @param dao octree mesh
  @param xPosArr x coordinates of points
  @param yPosArr y coordinates of points
  @param zPosArr z coordinates of points
  @param outOfBndsList indices of points outside local domain
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param F image
  @param gradF gradient of image
  @param results result
  */
void newEvalGradFnAtAllPts(ot::DA* dao, const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    const std::vector<unsigned int>& outOfBndsList,
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& F, const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

/**
  @brief cubic interpolation of the function
  @param fArr image values 
  @param gArr gradient of image values
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param xPos x coordinate
  @param yPos y coordinate
  @param zPos z coordinate
  */
double newEvalCubicFn(const std::vector<double>& fArr, 
    const std::vector<double>& gArr, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, double xPos, double yPos, double zPos);

/**
  @brief cubic interpolation of the gradient of function
  @param fArr image values 
  @param gArr gradient of image values
  @param imgPatches local image copy
  @param mesh mesh info for local image copy
  @param xPos x coordinate
  @param yPos y coordinate
  @param zPos z coordinate
  @param res result
  */
void newEvalCubicGradFn(const std::vector<double>& fArr,
    const std::vector<double>& gArr, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, double xPos, double yPos, double zPos, double* res);

/**
  @brief Evaluates the 32 cubic polynomials at the given point
  @param psi local coordinate of the point
  @param eta local coordinate of the point
  @param gamma local coordinate of the point
  @param phiArr result
  */
void evalAll3Dcubic(double psi, double eta, double gamma, double phiArr[8][4]);

/**
  @brief Evaluates the gradients of the 32 cubic polynomials at the given point
  @param psi local coordinate of the point
  @param eta local coordinate of the point
  @param gamma local coordinate of the point
  @param gradPhiArr result
  */
void evalAll3DcubicGrad(double psi, double eta, double gamma, double gradPhiArr[8][4][3]);

/**
  @brief 1-D cubic polynomial
  */
double eval1Dcubic(int nodeNum, int compNum, double psi);

/**
  @brief  gradient of 1-D cubic polynomial
  */
double eval1DcubicGrad(int nodeNum, int compNum, double psi);

/**
  @brief interpolate the gradient of moving image at mesh points (Deprecated) 
  */
void computeNodalGradTauAtU(ot::DA* dao, int Ne,
    const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalGradTauAtU);

/**
  @brief interpolate the moving image at mesh points (Deprecated) 
  */
void computeNodalTauAtU(ot::DA* dao, int Ne, 
    const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalTauAtU);

/**
  @brief interpolate the gradient of the moving image at gauss points (Deprecated) 
  */
void computeGradTauAtU(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU); 

/**
  @brief interpolate the moving image at gauss points (Deprecated) 
  */
void computeTauAtU(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU);

/**
  @brief interpolate the fixed image at gauss points (Deprecated) 
  */
void computeSigVals(ot::DA* dao, int Ne, const std::vector<std::vector<double> >& sig,
    const std::vector<std::vector<double> >& gradSig, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals);

/**
  @brief interpolate the moving image at regular grid mesh points (Deprecated) 
  */
void computeRegularNodalTauAtU(int Ne, const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalTauAtU);

/**
  @brief interpolate the image at given points (Deprecated) 
  */
void evalFnAtAllPts(const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    int Ne, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

/**
  @brief interpolate the gradient image at given points (Deprecated) 
  */
void evalGradFnAtAllPts(const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    int Ne, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

/**
  @brief cubic interpolation of the function (Deprecated) 
  */
double evalCubicFn(const double* fArr, const double* gArr, int Ne,
    double xPos, double yPos, double zPos);

/**
  @brief cubic interpolation of the gradient of the function (Deprecated) 
  */
void evalCubicGradFn(const double* fArr, const double* gArr, int Ne,
    double xPos, double yPos, double zPos, double* res);

#endif 



