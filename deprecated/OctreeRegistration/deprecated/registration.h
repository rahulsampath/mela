
/**
 * @file registration.h
 * @author	Rahul S. Sampath, rahul.sampath@gmail.com
 * 
**/ 

#ifndef _REGISTRATION_H_
#define _REGISTRATION_H_

#include "oda.h"
#include "omg.h"
#include "petscda.h"
#include "dbh.h"

struct HessData {
  unsigned char* bdyArr;
  std::vector<double>* gtVec;
  std::vector<ot::TreeNode>* imgPatches;
  std::vector<unsigned int>* mesh;
  std::vector<std::vector<double> >* tau; 
  std::vector<std::vector<double> >* gradTau; 
  std::vector<std::vector<double> >* sigVals;
  std::vector<std::vector<double> >* tauAtU; 
  std::vector<std::vector<double> >* gradTauAtU; 
  Mat Jmat_private;
  Vec inTmp;
  Vec outTmp;
  Vec U;
  Vec uTmp;
  int Ne;
  double patchWidth;
  double mu;
  double lambda;
  double alpha;
  int numGpts;
  double* gWts; 
  double* gPts;
  double****** PhiMatStencil;
  double**** LaplacianStencil; 
  double**** GradDivStencil; 
};

void copyValuesToImagePatches(DA dar,
    const std::vector<ot::TreeNode>& imgPatches,
    const std::vector<std::vector<double> >& sigGlobal,
    const std::vector<std::vector<double> >& gradSigGlobal,
    const std::vector<std::vector<double> >& tauGlobal, 
    const std::vector<std::vector<double> >& gradTauGlobal,
    std::vector<std::vector<double> >& sigLocal,
    std::vector<std::vector<double> >& gradSigLocal,
    std::vector<std::vector<double> >& tauLocal,
    std::vector<std::vector<double> >& gradTauLocal);

void meshImagePatches(std::vector<ot::TreeNode>& imgPatches,
    std::vector<unsigned int>& mesh);

void expandImagePatches(int Ne, double width, const std::vector<ot::TreeNode>& blocks,
    std::vector<ot::TreeNode>& imgPatches);

void createImagePatches(int Ne, ot::DA* da, std::vector<ot::TreeNode>& imgPatches);

bool foundValidDApart(int N, int npes);

void mortonToRgGlobalDisp(ot::DAMG* damg, DA dar, int Ne,
    std::vector<double> & dispOct, Vec & UrgGlobal);

void newProcessImgNatural(DA da1dof, DA da3dof, int Ne, Vec sigNatural, Vec tauNatural,
    std::vector<std::vector<double> > &  sigGlobal, std::vector<std::vector<double> > & gradSigGlobal,
    std::vector<std::vector<double> > & tauGlobal, std::vector<std::vector<double> > & gradTauGlobal,
    std::vector<double> & sigElemental, std::vector<double> & tauElemental);

void processImgNatural(DA da1dof, DA da3dof, int Ne, Vec sigNatural, Vec tauNatural,
    std::vector<std::vector<double> > &  sig, std::vector<std::vector<double> > & gradSig,
    std::vector<std::vector<double> > & tau, std::vector<std::vector<double> > & gradTau,
    std::vector<double> & sigElemental, std::vector<double> & tauElemental);

void createImgNodalNatural(DA da, const std::vector<double>& sigElemImg,
    const std::vector<double>& tauElemImg, Vec & sigNatural, Vec & tauNatural);

void createImgN0ToNatural(DA da, Vec sigN0, Vec tauN0,
    Vec & sigNatural, Vec & tauNatural, MPI_Comm comm);

void createSeqNodalImageVec(int Ne, int rank, int npes, const std::vector<double>& img,
    Vec & imgN0, MPI_Comm comm);

void coarsenPrlImage(DA daf, DA dac, bool cActive, const std::vector<double>& imgF, std::vector<double>& imgC);

void coarsenImage(int Nfe, const std::vector<double>& imgF, std::vector<double>& imgC);

void readImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> & img);

void writeImage(char* fnamePrefix, int nx, int ny, int nz, std::vector<double> & img);

void createCheckerBoardImage(int Ne, int width, Vec v);

void createFineAnalyticFixedImage(int Ne, int width, Vec v);

void createFineAnalyticImage(int Ne, int width, Vec v);

void createAnalyticImage(int Ne, int width, Vec v);

void createSphereImage(int Ne, int width, Vec v);

void createCshapeImage(int Ne, int width, Vec v);

void detJacMaxAndMin(DA da,  Vec u, double* maxDetJac,  double* minDetJac);

void saveVector(Vec v, char* fname);

void loadSeqVector(Vec & v, char* fname);

void loadPrlVector(Vec & v, char* fname);

double evalGuass3D(double x0, double y0, double z0,
    double sx, double sy, double sz,
    double x, double y, double z); 

void newGaussNewton(ot::DAMG* damg, double fTol, double xTol, int maxIterCnt, double patchWidth, Vec Uin, Vec Uout);

void gaussNewton(ot::DAMG* damg, Vec Uin, Vec Uout);

double evalPhi(int cNum, int eType, int nodeNum, double psi, double eta, double gamma);

double evalObjective(ot::DA* da, const std::vector<std::vector<double> >& sigVals,
    const std::vector<std::vector<double> >& tauAtU, int numGpts, double* gWts,
    unsigned char* bdyArr, double**** LaplacianStencil, 
    double**** GradDivStencil, double mu, double lambda, double alpha, Vec U, Vec tmp);

double computeObjImgPart(ot::DA* da, const std::vector<std::vector<double> >& sigVals,
    const std::vector<std::vector<double> >& tauAtU, int numGpts, double* gWts);

void computeGradientImgPart(ot::DA* da, const std::vector<std::vector<double> >& sigVals,
    const std::vector<std::vector<double> >& tauAtU,
    const std::vector<std::vector<double> >& gradTauAtU, 
    unsigned char* bdyArr, double****** PhiMatStencil,
    int numGpts, double* gWts, Vec g);

PetscErrorCode elasMatVec(ot::DA* da, unsigned char* bdyArr,
    double**** LaplacianStencil, double**** GradDivStencil,
    double mu, double lambda, Vec in, Vec out);

void enforceBC(ot::DA* da, unsigned char* bdyArr, Vec U);

PetscErrorCode computeDummyRHS(ot::DAMG damg, Vec rhs);

PetscErrorCode computeRHS(ot::DAMG damg, Vec rhs);

PetscErrorCode computeGaussianRHS(ot::DAMG damg, Vec rhs);

void zeroDouble(double & v);

void createNewHessContexts(ot::DAMG* damg, double patchWidth, std::vector<ot::TreeNode>& imgPatches,
    std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& sig,
    const std::vector<std::vector<double> >& gradSig,
    const std::vector<std::vector<double> >& tau, const std::vector<std::vector<double> >& gradTau,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts,
    double mu, double lambda, double alpha);

PetscErrorCode updateNewHessContexts(ot::DAMG* damg, Vec U);

void createHessContexts(ot::DAMG* damg, int Ne,
    const std::vector<std::vector<double> >& sig, const std::vector<std::vector<double> >& gradSig,
    const std::vector<std::vector<double> >& tau, const std::vector<std::vector<double> >& gradTau,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts,
    double mu, double lambda, double alpha);

PetscErrorCode updateHessContexts(ot::DAMG* damg, Vec U);

void destroyHessContexts(ot::DAMG* damg);

PetscErrorCode computeHessMat(ot::DAMG damg, Mat J, Mat B);

PetscErrorCode hessShellMatMult(Mat J, Vec in, Vec out);

PetscErrorCode hessMatMult(Mat J, Vec in, Vec out);

PetscErrorCode createHessMat(ot::DAMG damg, Mat *jac);

PetscErrorCode hessMatDestroy(Mat J);

void computeInvBlockDiagEntriesForHessMat(Mat J, double **invBlockDiagEntries);

void getDofAndNodeSizeForHessMat(Mat J, unsigned int & dof, unsigned int & nodeSize);

void getActiveStateAndActiveCommForKSP_Shell_Hess(Mat mat,
    bool & activeState, MPI_Comm & activeComm);

void getPrivateMatricesForKSP_Shell_Hess(Mat mat,
    Mat *AmatPrivate, Mat *PmatPrivate, MatStructure* pFlag);

void computeLumpedCoeffs(ot::DA* da, unsigned char* bdyArr,
    const std::vector<std::vector<double> >& gradTauAtU,
    double****** PhiMatStencil, double**** MassMatStencil, int numGpts,
    double* gWts, std::vector<double>& gt11,
    std::vector<double>& gt12, std::vector<double>& gt13,
    std::vector<double>& gt22, std::vector<double>& gt23, std::vector<double>& gt33);

void computeFDgradient(DA dai, DA dao, Vec in, Vec out);

void computeFDhessian(DA dai, DA dao, Vec in, Vec out);

void newComposeXposAtUnodal(ot::DA* dao, double patchWidth, Vec U, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr, 
    std::vector<unsigned int>& outOfBndsList);

void composeXposAtUnodal(ot::DA* dao, Vec U, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr);

void composeXposAtU(ot::DA* dao, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr);

void getSeqRGdisp(ot::DA* dao, Vec Uoct, int Ne, Vec & Urg);

int createLmat(double****& Lmat);
int createGDmat(double****& GDmat);
int createPhimat(double******& Phimat, int numGpts, double* gPts);

int createLmat_Type1(double****& Lmat);
int createGDmat_Type1(double****& GDmat);

int createLmat_Type2(double****& Lmat);
int createGDmat_Type2(double****& GDmat);

int createLmat_Type3(double****& Lmat);
int createGDmat_Type3(double****& GDmat);

int destroyLmat(double****& Lmat);
int destroyGDmat(double****& GDmat);
int destroyPhimat(double******& Phimat, int numGpts);

#endif /*REGISTRATION_H_*/



