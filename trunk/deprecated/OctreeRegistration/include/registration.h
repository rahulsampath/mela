
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

/**
  @brief context for the hessian matrix
  */
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

/**
  @brief scatter image values from PETSC DA global ordering to octree based (local) ordering
  @param dar PETSC DA
  @param imgPatches local image data structure
  @param sigGlobal fixed image in PETSC ordering
  @param gradSigGlobal gradient of fixed image in PETSC ordering
  @param tauGlobal moving image in PETSC ordering
  @param gradTauGlobal gradient of moving image in PETSC ordering
  @param sigLocal fixed image in local ordering
  @param gradSigLocal gradient of fixed image in local ordering
  @param tauLocal moving image in local ordering
  @param gradTauLocal gradient of moving image in local ordering
  */
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

/**
  @brief create mesh data structure for managing the local image data
  @param imgPatches local image data structure
  @param mesh associated mesh data structure
  */
void meshImagePatches(std::vector<ot::TreeNode>& imgPatches,
    std::vector<unsigned int>& mesh);

/**
  @brief Step-2 of creating image patches. Call after calling 'createImagePatches'
  @param Ne global number of elements in regular grid in each dimension
  @param width thickness of the image buffer zone
  @param blocks this is got by calling 'ot::DA::getBlocks()'
  @param imgPatches local image data strucuture
  */
void expandImagePatches(int Ne, double width, const std::vector<ot::TreeNode>& blocks,
    std::vector<ot::TreeNode>& imgPatches);

/**
  @brief Step-1 of creating image patches. 
  @param Ne global number of elements in regular grid in each dimension
  @param da octree DA
  @param imgPatches local image data strucuture
  */
void createImagePatches(int Ne, ot::DA* da, std::vector<ot::TreeNode>& imgPatches);

/**
  @brief Check if a valid PETSC DA partition can be found
  @param N number of node in DA in each direction
  @param npes number of processors
  */
bool foundValidDApart(int N, int npes);

/**
  @brief convert displacement field from octree mesh points to regular grid mesh points
  @param damg MG object
  @param dar PETSC DA  
  @param Ne number of elements in PETSC DA in each direction
  @param dispOct displacements at octree mesh points
  @param UrgGlobal displacements at regular grid mesh points
  */
void mortonToRgGlobalDisp(ot::DAMG* damg, DA dar, int Ne,
    std::vector<double> & dispOct, Vec & UrgGlobal);

/**
  @brief Scatter images from PETSC DA natural ordering to PETSC DA global ordering
  @param da1dof PETSC DA with 1 degree of freedom (dof) per node
  @param da3dof PETSC DA with 3 dof per node
  @param Ne number of elements in PETSC DA in each direction
  @param sigNatural fixed image in PETSC DA Natural ordering
  @param tauNatural moving image in PETSC DA Natural ordering
  @param sigGlobal fixed image in PETSC DA Global ordering
  @param tauGlobal moving image in PETSC DA Global ordering
  @param gradSigGlobal gradient of fixed image in PETSC DA Global ordering
  @param gradTauGlobal gradient of moving image in PETSC DA Global ordering
  @param sigElemental fixed image values for each element 
  @param tauElemental moving image values for each element
  */
void newProcessImgNatural(DA da1dof, DA da3dof, int Ne, Vec sigNatural, Vec tauNatural,
    std::vector<std::vector<double> > &  sigGlobal, std::vector<std::vector<double> > & gradSigGlobal,
    std::vector<std::vector<double> > & tauGlobal, std::vector<std::vector<double> > & gradTauGlobal,
    std::vector<double> & sigElemental, std::vector<double> & tauElemental);

/**
  @brief elemental to nodal 
  @param da PETSC DA
  @param sigElemImg elemental fixed image
  @param tauElemImg elemental moving image
  @param sigNatural fixed image in PETSC DA Natural ordering
  @param tauNatural moving image in PETSC DA Natural ordering
  */
void createImgNodalNatural(DA da, const std::vector<double>& sigElemImg,
    const std::vector<double>& tauElemImg, Vec & sigNatural, Vec & tauNatural);

/**
  @brief sequential to distributed image
  @param da PETSC DA
  @param sigN0 sequential fixed image
  @param tauN0 sequential moving image
  @param sigNatural fixed image in PETSC DA Natural ordering
  @param tauNatural moving image in PETSC DA Natural ordering
  */
void createImgN0ToNatural(DA da, Vec sigN0, Vec tauN0,
    Vec & sigNatural, Vec & tauNatural, MPI_Comm comm);

void createSeqNodalImageVec(int Ne, int rank, int npes, const std::vector<double>& img,
    Vec & imgN0, MPI_Comm comm);

/**
  @brief coarsen image (parallel)
  @param daf fine grid PETSC DA
  @param dac coarse grid PETSC DA
  @param cActive true iff coarse grid is active on calling processor
  @param imgF fine grid image
  @param imgC coarse grid image
  */
void coarsenPrlImage(DA daf, DA dac, bool cActive, const std::vector<double>& imgF, std::vector<double>& imgC);

/**
  @brief coarsen image (sequential)
  @param Nfe number of fine grid elements per dimension
  @param imgF fine image vector
  @param imgC coarse image vector
  */
void coarsenImage(int Nfe, const std::vector<double>& imgF, std::vector<double>& imgC);

void readImage(char* fnamePrefix, struct dsr* hdr, std::vector<double> & img);

void writeImage(char* fnamePrefix, int nx, int ny, int nz, std::vector<double> & img);

void createCheckerBoardImage(int Ne, int width, Vec v);

void createFineAnalyticFixedImage(int Ne, int width, Vec v);

void createFineAnalyticImage(int Ne, int width, Vec v);

void createAnalyticImage(int Ne, int width, Vec v);

void createSphereImage(int Ne, int width, Vec v);

void createCshapeImage(int Ne, int width, Vec v);

/**
  @brief computes the determinant of jacobian of the deformation
  */
void detJacMaxAndMin(DA da,  Vec u, double* maxDetJac,  double* minDetJac);

void saveVector(Vec v, char* fname);

void loadSeqVector(Vec & v, char* fname);

void loadPrlVector(Vec & v, char* fname);

/**
  @brief evaluates the 3-D gaussian function at the given point
  */
double evalGuass3D(double x0, double y0, double z0,
    double sx, double sy, double sz,
    double x, double y, double z); 

void newGaussNewton(ot::DAMG* damg, double fTol, double xTol, int maxIterCnt, double patchWidth, Vec Uin, Vec Uout);

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

/**
  @brief Elasticity Matvec
  */
PetscErrorCode elasMatVec(ot::DA* da, unsigned char* bdyArr,
    double**** LaplacianStencil, double**** GradDivStencil,
    double mu, double lambda, Vec in, Vec out);

/**
  @brief Zeros out values at boundaries
  */
void enforceBC(ot::DA* da, unsigned char* bdyArr, Vec U);

PetscErrorCode computeDummyRHS(ot::DAMG damg, Vec rhs);

/**
  @brief RHS for Newton step (-ve gradient of the objective)
  */
PetscErrorCode computeRHS(ot::DAMG damg, Vec rhs);

PetscErrorCode computeGaussianRHS(ot::DAMG damg, Vec rhs);

/**
  @brief sets v = 0
  */
void zeroDouble(double & v);

void createNewHessContexts(ot::DAMG* damg, double patchWidth, std::vector<ot::TreeNode>& imgPatches,
    std::vector<unsigned int>& mesh, const std::vector<std::vector<double> >& sig,
    const std::vector<std::vector<double> >& gradSig,
    const std::vector<std::vector<double> >& tau, const std::vector<std::vector<double> >& gradTau,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts,
    double mu, double lambda, double alpha);

PetscErrorCode updateNewHessContexts(ot::DAMG* damg, Vec U);

void destroyHessContexts(ot::DAMG* damg);

/**
  (Deprecated)
  */
void createHessContexts(ot::DAMG* damg, int Ne,
    const std::vector<std::vector<double> >& sig, const std::vector<std::vector<double> >& gradSig,
    const std::vector<std::vector<double> >& tau, const std::vector<std::vector<double> >& gradTau,
    double****** PhiMatStencil, double**** LaplacianStencil, double**** GradDivStencil, 
    int numGpts, double* gWts, double* gPts,
    double mu, double lambda, double alpha);

/**
  (Deprecated)
  */
PetscErrorCode updateHessContexts(ot::DAMG* damg, Vec U);

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

/**
  @brief computes the gradient of the image using second order central differencing
  @param dai PETSC DA with 1 dof (for image)
  @param dao PETSC DA with 3 dof (for gradient)
  @param in image
  @param out gradient of the image
  */
void computeFDgradient(DA dai, DA dao, Vec in, Vec out);

void computeFDhessian(DA dai, DA dao, Vec in, Vec out);

/**
  @brief Evaluates x + u(x) at mesh points
  @param dao Octree DA
  @param patchwidth thickness of image buffer layer
  @param U displacements at mesh points
  @param xPosArr x coordinates of the resulting points
  @param yPosArr y coordinates of the resulting points
  @param zPosArr z coordinates of the resulting points
  @param outOfBndsList indices of points that need to be
  communicated to other processors since they are outside
  the region controlled by this processor  
  */
void newComposeXposAtUnodal(ot::DA* dao, double patchWidth, Vec U, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr, 
    std::vector<unsigned int>& outOfBndsList);

/**
  (Deprecated)
  */
void composeXposAtUnodal(ot::DA* dao, Vec U, std::vector<double>& xPosArr,
    std::vector<double>& yPosArr, std::vector<double>& zPosArr);

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



