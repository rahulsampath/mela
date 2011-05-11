
#ifndef _REG_INTERP_CUBIC_H_
#define _REG_INTERP_CUBIC_H_

#include "oda.h"
#include "petscda.h"

void computeNewGradTauAtU(ot::DA* dao, double patchWidth, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU); 

void computeNewTauAtU(ot::DA* dao, double patchWidth, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU);

void computeNewSigVals(ot::DA* dao, double patchWidth, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& sigLocal,
    const std::vector<std::vector<double> >& gradSigLocal, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals);

void computeNewNodalGradTauAtU(ot::DA* dao, double patchWidth,
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal,
    Vec U, std::vector<std::vector<double> >& nodalGradTauAtU);

void computeNewNodalTauAtU(ot::DA* dao, double patchWidth, 
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& tauLocal,
    const std::vector<std::vector<double> >& gradTauLocal,
    Vec U, std::vector<std::vector<double> >& nodalTauAtU);

void newEvalFnAtAllPts(ot::DA* dao, const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    const std::vector<unsigned int>& outOfBndsList,
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& F, const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

void newEvalGradFnAtAllPts(ot::DA* dao, const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    const std::vector<unsigned int>& outOfBndsList,
    const std::vector<ot::TreeNode>& imgPatches, const std::vector<unsigned int>& mesh,
    const std::vector<std::vector<double> >& F, const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

double newEvalCubicFn(const std::vector<double>& fArr, 
    const std::vector<double>& gArr, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, double xPos, double yPos, double zPos);

void newEvalCubicGradFn(const std::vector<double>& fArr,
    const std::vector<double>& gArr, const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<unsigned int>& mesh, double xPos, double yPos, double zPos, double* res);

void computeNodalGradTauAtU(ot::DA* dao, int Ne,
    const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalGradTauAtU);

void computeNodalTauAtU(ot::DA* dao, int Ne, 
    const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalTauAtU);

void computeGradTauAtU(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU); 

void computeTauAtU(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU);

void computeSigVals(ot::DA* dao, int Ne, const std::vector<std::vector<double> >& sig,
    const std::vector<std::vector<double> >& gradSig, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals);

void computeRegularNodalTauAtU(int Ne, const std::vector<std::vector<double> >& tau,
    const std::vector<std::vector<double> >& gradTau, Vec U,
    std::vector<std::vector<double> >& nodalTauAtU);

void computeGradTauAtU_Old(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU); 

void computeTauAtU_Old(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU);

void computeSigVals_Old(ot::DA* dao, int Ne, const std::vector<std::vector<double> >& sig,
    const std::vector<std::vector<double> >& gradSig, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals);

void evalFnAtAllPts(const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    int Ne, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

void evalGradFnAtAllPts(const std::vector<double>& xPosArr,
    const std::vector<double>& yPosArr, const std::vector<double>& zPosArr,
    int Ne, const std::vector<std::vector<double> >& F,
    const std::vector<std::vector<double> >& gradF,
    std::vector<std::vector<double> >& results);

double evalCubicFn(const double* fArr, const double* gArr, int Ne,
    double xPos, double yPos, double zPos);

void evalCubicGradFn(const double* fArr, const double* gArr, int Ne,
    double xPos, double yPos, double zPos, double* res);

void evalAll3Dcubic(double psi, double eta, double gamma, double phiArr[8][4]);

void evalAll3DcubicGrad(double psi, double eta, double gamma, double gradPhiArr[8][4][3]);

double eval1Dcubic(int nodeNum, int compNum, double psi);

double eval1DcubicGrad(int nodeNum, int compNum, double psi);

#endif 



