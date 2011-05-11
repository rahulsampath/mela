
#ifndef _REG_INTERP_QUINTIC_H_
#define _REG_INTERP_QUINTIC_H_

#include "oda.h"
#include "petscda.h"

void evalQuinticGradFnAtAllPts(std::vector<double>& xPosArr, std::vector<double>& yPosArr,
    std::vector<double>& zPosArr, DA dar1, DA dar2, DA dar3,
    Vec fnLoc, Vec gradLoc, Vec hessLoc, std::vector<double>& results, MPI_Comm comm);

void evalQuinticFnAtAllPts(std::vector<double>& xPosArr, std::vector<double>& yPosArr,
    std::vector<double>& zPosArr, DA dar1, DA dar2, DA dar3,
    Vec fnLoc, Vec gradLoc, Vec hessLoc, std::vector<double>& results, MPI_Comm comm);

double evalQuinticFn(const PetscScalar*** fArr, const PetscScalar**** gArr, 
    const PetscScalar**** hArr, double xOff, double yOff, double zOff,
    double h, double xPos, double yPos, double zPos);

void evalQuinticGradFn(const PetscScalar*** fArr, const PetscScalar**** gArr, 
    const PetscScalar**** hArr, double xOff, double yOff, double zOff,
    double h, double xPos, double yPos, double zPos, double* res);

double eval1DquinticGrad(int nodeNum, int compNum, double psi);

double eval1Dquintic(int nodeNum, int compNum, double psi);

void evalAll3DquinticGrad(double psi, double eta, double gamma, double*** gradPhiArr);

void evalAll3Dquintic(double psi, double eta, double gamma, double** phiArr);

void computeQuinticSigVals(ot::DA* dao, DA dar1, DA dar2, DA dar3,
    Vec sigLoc, Vec gradSigLoc, Vec hessSigLoc,
    int numGpts, const double* gPts, std::vector<double>& sigVals);

void computeQuinticTauAtU(ot::DA* dao, DA dar1, DA dar2, DA dar3,
    Vec tauLoc, Vec gradTauLoc, Vec hessTauLoc, Vec U,
    const double****** PhiMatStencil, int numGpts,
    const double* gPts, std::vector<double>& tauAtU);

void computeQuinticGradTauAtU(ot::DA* dao, DA dar1, DA dar2, DA dar3,
    Vec tauLoc, Vec gradTauLoc, Vec hessTauLoc, Vec U,
    const double****** PhiMatStencil, int numGpts,
    const double* gPts, std::vector<double>& gradTauAtU);

#endif 



