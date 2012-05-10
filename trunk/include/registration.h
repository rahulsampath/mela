
/**
 * @file registration.h
 * @author	Rahul S. Sampath, rahul.sampath@gmail.com
 * 
**/ 

#ifndef _REGISTRATION_H_
#define _REGISTRATION_H_

#include "mpi.h"
#include "oda/oda.h"
#include "omg/omg.h"
#include "petscda.h"
#include <vector>
#include "dbh.h"

struct HessData {
  unsigned char* bdyArr;
  std::vector<double>* gtVec;
  std::vector<std::vector<double> >* tauLocal; 
  std::vector<std::vector<double> >* gradTauLocal; 
  std::vector<double>* sigVals;
  std::vector<double>* tauAtU; 
  std::vector<double>* gradTauAtU; 
  Mat Jmat_private;
  Vec inTmp;
  Vec outTmp;
  Vec U;
  Vec uTmp;
  int Ne;
  int padding;
  int numImages;
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

void processImgNatural(DA da1dof, DA da3dof, Vec sigNatural, Vec tauNatural,
    std::vector<double> &  sigGlobal, std::vector<double> & gradSigGlobal,
    std::vector<double> & tauGlobal, std::vector<double> & gradTauGlobal,
    std::vector<double> & sigElemental, std::vector<double> & tauElemental);

void createImgNodalNatural(DA da, const std::vector<double>& sigElemImg,
    const std::vector<double>& tauElemImg, Vec & sigNatural, Vec & tauNatural);

void createSeqNodalImageVec(int Ne, int numImages, int rank, int npes, const std::vector<double>& img,
    Vec & imgN0, MPI_Comm comm);

void createImgN0ToNatural(DA da, Vec sigN0, Vec tauN0,
    Vec & sigNatural, Vec & tauNatural, MPI_Comm comm);

double gaussNewton(ot::DAMG* damg, double fTol, double xTol, int maxIterCnt, Vec Uin, Vec Uout);

double evalGauss3D(double x0, double y0, double z0,
    double sx, double sy, double sz,
    double x, double y, double z); 

PetscErrorCode computeRHS(ot::DAMG damg, Vec rhs);

double evalObjective(ot::DA* da, const std::vector<double> & sigVals,
    const std::vector<double> & tauAtU, int numImages, int numGpts, double* gWts,
    unsigned char* bdyArr, double**** LaplacianStencil, 
    double**** GradDivStencil, double mu, double lambda, double alpha, Vec U, Vec tmp);

double computeObjImgPart(ot::DA* da, const std::vector<double>& sigVals,
    const std::vector<double> & tauAtU, int numImages, int numGpts, double* gWts);

void computeGradientImgPart(ot::DA* da, const std::vector<double> & sigVals,
    const std::vector<double> & tauAtU, const std::vector<double> & gradTauAtU, 
    unsigned char* bdyArr, double****** PhiMatStencil, int numImages, 
    int numGpts, double* gWts, Vec g);

void createHessContexts(ot::DAMG* damg, int Ne, int padding, int numImages,
    const std::vector<std::vector<double> >& sigLocal,
    const std::vector<std::vector<double> >& gradSigLocal,
    const std::vector<std::vector<double> >& tauLocal, 
    const std::vector<std::vector<double> >& gradTauLocal,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts, double mu, double lambda, double alpha);

void updateHessContexts(ot::DAMG* damg, Vec U);

void destroyHessContexts(ot::DAMG* damg);

PetscErrorCode elasMatVec(ot::DA* da, unsigned char* bdyArr,
    double**** LaplacianStencil, double**** GradDivStencil,
    double mu, double lambda, Vec in, Vec out);

PetscErrorCode computeHessMat(ot::DAMG damg, Mat J, Mat B);

PetscErrorCode hessShellMatMult(Mat J, Vec in, Vec out);

PetscErrorCode hessMatMult(Mat J, Vec in, Vec out);

PetscErrorCode createHessMat(ot::DAMG damg, Mat *jac);

PetscErrorCode hessMatDestroy(Mat J);

void computeInvBlockDiagEntriesForHessMat(Mat J, double **invBlockDiagEntries);

/**
  @param dar PETSC DA
  @param dao Dendro DA
  @param padding Thickness (number of voxels) of the buffer zone
  @param numImages Number of features in the images
  @param sigGlobal Distributed fixed image in PETSC's global ordering
  @param tauGlobal Distributed moving image in PETSC's global ordering
  @param gradSigGlobal Distributed fixed image's gradient in PETSC's global ordering
  @param gradTauGlobal Distributed moving image's gradient in PETSc's global ordering
  @param sigLocal local copy of fixed image aligned with octree
  @param tauLocal local copy of moving image aligned with octree
  */
void createImagePatches(DA dar, ot::DA* dao, int padding, int numImages,
    const std::vector<double >& sigGlobal, 
    const std::vector<double >& tauGlobal,
    const std::vector<double >& gradSigGlobal,
    const std::vector<double >& gradTauGlobal,
    std::vector<std::vector<double> >& sigLocal,
    std::vector<std::vector<double> >& tauLocal,
    std::vector<std::vector<double> >& gradSigLocal,
    std::vector<std::vector<double> >& gradTauLocal);

void getDofAndNodeSizeForHessMat(Mat J, unsigned int & dof, unsigned int & nodeSize);

void getActiveStateAndActiveCommForKSP_Shell_Hess(Mat mat,
    bool & activeState, MPI_Comm & activeComm);

void getPrivateMatricesForKSP_Shell_Hess(Mat mat,
    Mat *AmatPrivate, Mat *PmatPrivate, MatStructure* pFlag);

void computeFDgradient(DA dai, DA dao, Vec in, Vec out);

void computeFDhessian(DA dai, DA dao, Vec in, Vec out);

void zeroDouble(double & v);

bool foundValidDApart(int N, int npes);

void detJacMaxAndMin(DA da,  Vec u, double* maxDetJac,  double* minDetJac);

void enforceBC(ot::DA* da, unsigned char* bdyArr, Vec U);

double evalPhi(int cNum, int eType, int nodeNum, double psi, double eta, double gamma);

void readImages(char* fnamePrefix, struct dsr* hdr, int numImages, std::vector<double> & img);

void readScalarImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> & img, int imgId);

void readImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> & img);

void readScalarImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> &  img);

void writeImage(char* fnamePrefix, int imgId, int nx, int ny, int nz, std::vector<double> & img);

void writeScalarImage(char* fnamePrefix, int nx, int ny, int nz, std::vector<double> & img,int imgId);

void writeVecImg(char *scalarImg,char *vecImg,int numImg);

void writeScalarImg(char *vecImg,char *scalarImg,int numImg);

void genDispImg(char *disp,double dispScale, char* templateImg,char *dispImg,int computeDetJacMat, int numImg);

void concatDispFile(char *elasDisp,int numProc, char *genDisp);
void concatDisp(char *dispFile1,char *dispFile2,char *dispTarget);
//void concatSaveVector(Vec v1, Vec v2, char* fname); 

void coarsenPrlImage(DA daf, DA dac, bool cActive, int numImages, const std::vector<double>& imgF, std::vector<double>& imgC);
void coarsenImage(int Nfe, int numImages, const std::vector<double>& imgF, std::vector<double>& imgC);

void saveVector(Vec v, char* fname);

void loadSeqVector(Vec & v, char* fname);

void loadPrlVector(Vec & v, char* fname);

void mortonToRgGlobalDisp(ot::DAMG* damg, DA dar, int Ne,
    std::vector<double> & dispOct, Vec & UrgGlobal);

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

void createGaussPtsAndWts(double*& gPts, double*& gWts, int numGpts);
void destroyGaussPtsAndWts(double*& gPts, double*& gWts);

//int  elasReg(int argc, char **argv);

#endif /*REGISTRATION_H_*/


