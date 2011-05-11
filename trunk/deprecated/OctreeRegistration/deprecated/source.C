
void computeLumpedCoeffs(ot::DA* da, unsigned char* bdyArr,
    const std::vector<std::vector<double> >& gradTauAtU,
    double****** PhiMatStencil, double**** MassMatStencil, int numGpts,
    double* gWts, std::vector<double>& gt11,
    std::vector<double>& gt12, std::vector<double>& gt13,
    std::vector<double>& gt22, std::vector<double>& gt23, std::vector<double>& gt33) {

  unsigned int numElems = da->getElementSize();
  unsigned int numImages = gradTauAtU.size();

  gt11.resize(64*numElems);
  gt12.resize(64*numElems);
  gt13.resize(64*numElems);
  gt22.resize(64*numElems);
  gt23.resize(64*numElems);
  gt33.resize(64*numElems);

  for(int i = 0; i < (64*numElems); i++) {
    gt11[i] = 0;
    gt12[i] = 0;
    gt13[i] = 0;
    gt22[i] = 0;
    gt23[i] = 0;
    gt33[i] = 0;
  }

  if(da->iAmActive()) {
    unsigned int elemCnt = 0;
    unsigned int ptsPerElem = numGpts*numGpts*numGpts;
    for(da->init<ot::DA_FLAGS::WRITABLE>();
        da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
        da->next<ot::DA_FLAGS::WRITABLE>()) {
      unsigned int idx = da->curr();
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType, hnMask, childNum);
      for(int k = 0; k < 8; k++) {
        if(!bdyArr[indices[k]]){
          for(int j = 0; j < 8; j++) {
            if(!bdyArr[indices[j]]){
              double elemInt11 = 0;
              double elemInt12 = 0;
              double elemInt13 = 0;
              double elemInt22 = 0;
              double elemInt23 = 0;
              double elemInt33 = 0;
              double massVal = MassMatStencil[childNum][elemType][k][j];
              int ptCtr = 0;
              for(int m = 0; m < numGpts; m++) {
                for(int n = 0; n < numGpts; n++) {
                  for(int p = 0; p < numGpts; p++) {
                    double phiKval = PhiMatStencil[childNum][elemType][k][m][n][p];
                    double phiJval = PhiMatStencil[childNum][elemType][j][m][n][p];
                    double fVal11 = 0;
                    double fVal12 = 0;
                    double fVal13 = 0;
                    double fVal22 = 0;
                    double fVal23 = 0;
                    double fVal33 = 0;
                    for(int i = 0; i < numImages; i++) {
                      double gradTau1Val = gradTauAtU[i][3*((ptsPerElem*elemCnt) + ptCtr)];
                      double gradTau2Val = gradTauAtU[i][(3*((ptsPerElem*elemCnt) + ptCtr)) + 1];
                      double gradTau3Val = gradTauAtU[i][(3*((ptsPerElem*elemCnt) + ptCtr)) + 2];
                      fVal11 += (gradTau1Val*gradTau1Val);
                      fVal12 += (gradTau1Val*gradTau2Val);
                      fVal13 += (gradTau1Val*gradTau3Val);
                      fVal22 += (gradTau2Val*gradTau2Val);
                      fVal23 += (gradTau2Val*gradTau3Val);
                      fVal33 += (gradTau3Val*gradTau3Val);
                    }
                    double intFactor = (gWts[m]*gWts[n]*gWts[p]*phiKval*phiJval);
                    elemInt11 += (fVal11*intFactor);
                    elemInt12 += (fVal12*intFactor);
                    elemInt13 += (fVal13*intFactor);
                    elemInt22 += (fVal22*intFactor);
                    elemInt23 += (fVal23*intFactor);
                    elemInt33 += (fVal33*intFactor);
                    ptCtr++; 
                  }
                }
              }
              gt11[(64*elemCnt) + (8*k) + j] = (elemInt11/massVal);
              gt12[(64*elemCnt) + (8*k) + j] = (elemInt12/massVal);
              gt13[(64*elemCnt) + (8*k) + j] = (elemInt13/massVal);
              gt22[(64*elemCnt) + (8*k) + j] = (elemInt22/massVal);
              gt23[(64*elemCnt) + (8*k) + j] = (elemInt23/massVal);
              gt33[(64*elemCnt) + (8*k) + j] = (elemInt33/massVal);
            }
          }
        }
      }
      elemCnt++;
    }
  }
}


void getSeqRGdisp(ot::DA* dao, Vec Uoct, int Ne, Vec & Urg) {

  int rank = dao->getRankAll();
  int npes = dao->getNpesAll();

  VecCreate(dao->getComm(), &Urg);
  if(!rank) {
    VecSetSizes(Urg, (3*(Ne + 1)*(Ne + 1)*(Ne + 1)), PETSC_DECIDE);
  } else {
    VecSetSizes(Urg, 0, PETSC_DECIDE);
  }
  if(npes == 1) {
    VecSetType(Urg, VECSEQ);
  } else {
    VecSetType(Urg, VECMPI);
  }
  VecZeroEntries(Urg);

  std::vector<double> pts;
  if(!rank) {
    double h = 1.0/static_cast<double>(Ne);
    for(int k = 0; k < Ne; k++) {
      for(int j = 0; j < Ne; j++) {
        for(int i = 0; i < Ne; i++) {
          pts.push_back(static_cast<double>(i)*h);
          pts.push_back(static_cast<double>(j)*h);
          pts.push_back(static_cast<double>(k)*h);
        }
      }
    }
  }

  Vec UrgTmp;
  VecCreate(dao->getComm(), &UrgTmp);
  if(!rank) {
    VecSetSizes(UrgTmp, (3*Ne*Ne*Ne), PETSC_DECIDE);
  } else {
    VecSetSizes(UrgTmp, 0, PETSC_DECIDE);
  }
  if(npes == 1) {
    VecSetType(UrgTmp, VECSEQ);
  } else {
    VecSetType(UrgTmp, VECMPI);
  } 

  ot::interpolateData(dao, Uoct, UrgTmp, NULL, 3, pts);

  if(!rank) {
    PetscScalar* arr;
    PetscScalar* tmpArr;
    VecGetArray(Urg, &arr);
    VecGetArray(UrgTmp, &tmpArr);

    for(int k = 0; k < Ne; k++) {
      for(int j = 0; j < Ne; j++) {
        for(int i = 0; i < Ne; i++) {
          for(int d = 0; d < 3; d++) {
            arr[(3*((((k*(Ne + 1)) + j)*(Ne + 1)) + i)) + d] = tmpArr[(3*((((k*Ne) + j)*Ne) + i)) + d];
          }
        }
      }
    }

    VecRestoreArray(Urg, &arr);
    VecRestoreArray(UrgTmp, &tmpArr);
  }

  VecDestroy(UrgTmp);
}

PetscErrorCode hessMatMult(Mat J, Vec in, Vec out)
{
  PetscFunctionBegin;

  PetscLogEventBegin(hessMultEvent, in, out, 0, 0);

  ot::DAMG damg;
  MatShellGetContext(J, (void**)(&damg));

  assert(damg != NULL);

  bool isFinestLevel = (damg->nlevels == 1);

  if(isFinestLevel) {
    PetscLogEventBegin(hessFinestMultEvent, in, out, 0, 0);
  }

  ot::DA* da = damg->da;
  assert(da != NULL);

  HessData* data = (static_cast<HessData*>(damg->user));
  assert(data != NULL);

  VecZeroEntries(out);
  unsigned int maxD;
  double hFac;
  if(da->iAmActive()) {
    maxD = da->getMaxDepth();
    hFac = 1.0/((double)(1u << (maxD-1)));
  }
  PetscScalar *outArr=NULL;
  PetscScalar *inArr=NULL;
  unsigned char* bdyArr = data->bdyArr;
  double**** LaplacianStencil = data->LaplacianStencil;
  double**** GradDivStencil = data->GradDivStencil;
  double mu = data->mu;
  double lambda = data->lambda;
  double alpha = data->alpha;
  double* gtArr;

  int sixMap[][3] = { {0, 1, 2},
    {1, 3, 4},
    {2, 4, 5} };

  assert((data->gtVec) != NULL);

  /*Nodal,Non-Ghosted,Read,6 dof*/
  da->vecGetBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);

  /*Nodal,Non-Ghosted,Read,3 dof*/
  da->vecGetBuffer(in, inArr, false, false, true, 3);

  /*Nodal,Non-Ghosted,Write,3 dof*/
  da->vecGetBuffer(out, outArr, false, false, false, 3);

  if(da->iAmActive()) {
    da->ReadFromGhostsBegin<PetscScalar>(inArr, 3);

#ifndef __NO_COMM_OVERLAP__
    /*Loop through All Independent Elements*/
    for(da->init<ot::DA_FLAGS::INDEPENDENT>();
        da->curr() < da->end<ot::DA_FLAGS::INDEPENDENT>();
        da->next<ot::DA_FLAGS::INDEPENDENT>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      double facImg = (h*h*h)/8.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0; k < 8; k++) {
          if(bdyArr[indices[k]]) {
            /*Dirichlet Node Row*/
            for(int dof = 0; dof < 3; dof++) {
              outArr[(3*indices[k]) + dof] =  alpha*inArr[(3*indices[k]) + dof];
            }/*end for dof*/
          } else {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (alpha*mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += (alpha*(mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
            for(int dofOut = 0; dofOut < 3; dofOut++) {
              for(int dofIn = 0; dofIn < 3; dofIn++) {
                outArr[(3*indices[k]) + dofOut] += (facImg*
                    gtArr[(6*indices[k]) + sixMap[dofOut][dofIn]]*
                    inArr[(3*indices[k]) + dofIn]);
              }/*end for dofIn*/
            }/*end for dofOut*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end independent*/
#endif

    da->ReadFromGhostsEnd<PetscScalar>(inArr);

#ifndef __NO_COMM_OVERLAP__
    /*Loop through All Dependent Elements,*/
    /*i.e. Elements which have atleast one*/
    /*vertex owned by this processor and at least one*/
    /*vertex not owned by this processor.*/
    for(da->init<ot::DA_FLAGS::DEPENDENT>();
        da->curr() < da->end<ot::DA_FLAGS::DEPENDENT>();
        da->next<ot::DA_FLAGS::DEPENDENT>()) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      double facImg = (h*h*h)/8.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(bdyArr[indices[k]]) {
            /*Dirichlet Node Row*/
            for(int dof = 0; dof < 3; dof++) {
              outArr[(3*indices[k]) + dof] =  alpha*inArr[(3*indices[k]) + dof];
            }/*end for dof*/
          } else {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (alpha*mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += (alpha*(mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
            for(int dofOut = 0; dofOut < 3; dofOut++) {
              for(int dofIn = 0; dofIn < 3; dofIn++) {
                outArr[(3*indices[k]) + dofOut] += (facImg*
                    gtArr[(6*indices[k]) + sixMap[dofOut][dofIn]]*
                    inArr[(3*indices[k]) + dofIn]);
              }/*end for dofIn*/
            }/*end for dofOut*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for dependent elems*/
#endif

#ifdef __NO_COMM_OVERLAP__
    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      double facImg = (h*h*h)/8.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(bdyArr[indices[k]]) {
            /*Dirichlet Node Row*/
            for(int dof = 0; dof < 3; dof++) {
              outArr[(3*indices[k]) + dof] =  alpha*inArr[(3*indices[k]) + dof];
            }/*end for dof*/
          } else {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (alpha*mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += (alpha*(mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
            for(int dofOut = 0; dofOut < 3; dofOut++) {
              for(int dofIn = 0; dofIn < 3; dofIn++) {
                outArr[(3*indices[k]) + dofOut] += (facImg*
                    gtArr[(6*indices[k]) + sixMap[dofOut][dofIn]]*
                    inArr[(3*indices[k]) + dofIn]);
              }/*end for dofIn*/
            }/*end for dofOut*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for ALL*/
#endif

  } /*end if active*/

  da->vecRestoreBuffer(in,inArr,false,false,true,3);

  da->vecRestoreBuffer(out,outArr,false,false,false,3);

  da->vecRestoreBuffer<double>(*(data->gtVec), gtArr, false, false, true, 6);

  /*2 IOP = 1 FLOP. Loop counters are included too.*/
  PetscLogFlops(6891*(da->getGhostedElementSize()));

  if(isFinestLevel) {
    PetscLogEventEnd(hessFinestMultEvent, in, out, 0, 0);
  }

  PetscLogEventEnd(hessMultEvent, in, out, 0, 0);

  PetscFunctionReturn(0);
}



PetscErrorCode elasMatVec(ot::DA* da, unsigned char* bdyArr,
    double**** LaplacianStencil, double**** GradDivStencil,
    double mu, double lambda, Vec in, Vec out) {
  PetscFunctionBegin;

  PetscLogEventBegin(elasMultEvent, 0, 0, 0, 0);

  VecZeroEntries(out);

  assert(da != NULL);

  unsigned int maxD;
  double hFac;
  if(da->iAmActive()) {
    maxD = da->getMaxDepth();
    hFac = 1.0/((double)(1u << (maxD-1)));
  }
  PetscScalar *outArr=NULL;
  PetscScalar *inArr=NULL;

  /*Nodal,Non-Ghosted,Read,3 dof*/
  da->vecGetBuffer(in, inArr, false, false, true, 3);

  /*Nodal,Non-Ghosted,Write,3 dof*/
  da->vecGetBuffer(out, outArr, false, false, false, 3);

  if(da->iAmActive()) {
    da->ReadFromGhostsBegin<PetscScalar>(inArr, 3);

#ifndef __NO_COMM_OVERLAP__
    /*Loop through All Independent Elements*/
    for(da->init<ot::DA_FLAGS::INDEPENDENT>();
        da->curr() < da->end<ot::DA_FLAGS::INDEPENDENT>();
        da->next<ot::DA_FLAGS::INDEPENDENT>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0; k < 8; k++) {
          if(!bdyArr[indices[k]]) {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += ((mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end independent*/
#endif

    da->ReadFromGhostsEnd<PetscScalar>(inArr);

#ifndef __NO_COMM_OVERLAP__

#ifdef __USE_W_LOOP__

    /*Loop through All Dependent Elements,*/
    /*i.e. Elements which have atleast one*/
    /*vertex owned by this processor and at least one*/
    /*vertex not owned by this processor.*/
    for(da->init<ot::DA_FLAGS::W_DEPENDENT>();
        da->curr() < da->end<ot::DA_FLAGS::W_DEPENDENT>();
        da->next<ot::DA_FLAGS::W_DEPENDENT>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(!bdyArr[indices[k]]) {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += ((mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for w_dependent elems*/

    da->WriteToGhostsBegin<PetscScalar>(outArr, 3);
    da->WriteToGhostsEnd<PetscScalar>(outArr, 3);

#else

    /*Loop through All Dependent Elements,*/
    /*i.e. Elements which have atleast one*/
    /*vertex owned by this processor and at least one*/
    /*vertex not owned by this processor.*/
    for(da->init<ot::DA_FLAGS::DEPENDENT>();
        da->curr() < da->end<ot::DA_FLAGS::DEPENDENT>();
        da->next<ot::DA_FLAGS::DEPENDENT>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(!bdyArr[indices[k]]) {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += ((mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for dependent elems*/

#endif

#endif

#ifdef __NO_COMM_OVERLAP__

#ifdef __USE_W_LOOP__

    for(da->init<ot::DA_FLAGS::WRITABLE>();
        da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
        da->next<ot::DA_FLAGS::WRITABLE>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(!bdyArr[indices[k]]) {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += ((mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for WRITABLE*/

    da->WriteToGhostsBegin<PetscScalar>(outArr, 3);
    da->WriteToGhostsEnd<PetscScalar>(outArr, 3);

#else

    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>() ) {
      unsigned int idx = da->curr();
      unsigned int lev = da->getLevel(idx);
      double h = hFac*(1u << (maxD - lev));
      double facElas = h/2.0;
      unsigned int indices[8];
      da->getNodeIndices(indices);
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        for(int k = 0;k < 8;k++) {
          if(!bdyArr[indices[k]]) {
            for(int j = 0; j < 8; j++) {
              /*Avoid Dirichlet Node Columns*/
              if(!(bdyArr[indices[j]])) {
                for(int dof = 0; dof < 3; dof++) {
                  outArr[(3*indices[k]) + dof] += (mu*facElas*
                      LaplacianStencil[childNum][elemType][k][j]
                      *inArr[(3*indices[j]) + dof]);
                }/*end for dof*/
                for(int dofOut = 0; dofOut < 3; dofOut++) {
                  for(int dofIn = 0; dofIn < 3; dofIn++) {
                    outArr[(3*indices[k]) + dofOut] += ((mu+lambda)*facElas*
                        (GradDivStencil[childNum][elemType][(3*k) + dofOut][(3*j) + dofIn])
                        *inArr[(3*indices[j]) + dofIn]);
                  }/*end for dofIn*/
                }/*end for dofOut*/
              }/*end if boundary*/
            }/*end for j*/
          }/*end if boundary*/
        }/*end for k*/
    } /*end loop for ALL*/

#endif

#endif
  } /*end if active*/

  da->vecRestoreBuffer(in, inArr, false, false, true, 3);

  da->vecRestoreBuffer(out, outArr, false, false, false, 3);

  PetscLogEventEnd(elasMultEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}


void gaussNewton(ot::DAMG* damg, Vec Uin, Vec Uout) { 

  int iterCnt = 0;
  PetscInt maxIterCnt = 10;

  PetscTruth useSteepestDescent;
  PetscOptionsHasName(0, "-useSteepestDescent", &useSteepestDescent);

  PetscTruth saveGradient;
  PetscOptionsHasName(0, "-saveGradient", &saveGradient);

  int nlevels = damg[0]->nlevels;
  ot::DA* daFinest = damg[nlevels - 1]->da;
  HessData* ctxFinest = static_cast<HessData*>(damg[nlevels - 1]->user);
  int Ne = ctxFinest->Ne;
  int numGpts = ctxFinest->numGpts;
  double* gPts = ctxFinest->gPts;
  double* gWts = ctxFinest->gWts;
  double mu = ctxFinest->mu;
  double lambda = ctxFinest->lambda;
  double alpha = ctxFinest->alpha;
  std::vector<std::vector<double> >* tau = ctxFinest->tau;
  std::vector<std::vector<double> >* sigVals = ctxFinest->sigVals;
  std::vector<std::vector<double> >* gradTau = ctxFinest->gradTau;
  std::vector<std::vector<double> >* tauAtU = ctxFinest->tauAtU;
  double****** PhiMatStencil = ctxFinest->PhiMatStencil;
  double**** LaplacianStencil = ctxFinest->LaplacianStencil;
  double**** GradDivStencil = ctxFinest->GradDivStencil;
  unsigned char* bdyArr = ctxFinest->bdyArr;
  Vec uTmp = ctxFinest->uTmp;

  double objVal = 0;
  double objValInit = 0;

  PetscReal fTol = 0.001;
  PetscReal xTol = 0.1;

  PetscOptionsGetInt(0, "-gnMaxIterCnt", &maxIterCnt, 0);

  PetscOptionsGetReal(0, "-gnFntol", &fTol, 0);
  PetscOptionsGetReal(0, "-gnXtol", &xTol, 0);

  int rank = daFinest->getRankAll();
  int npes = daFinest->getNpesAll();

  //1. update HessContext
  updateHessContexts(damg, Uin);

  //2. Set new operators
  if(!useSteepestDescent) {
    ot::DAMGSetKSP(damg, createHessMat,  computeHessMat, computeRHS);
  }

  //3. Initial Objective
  objValInit = evalObjective(daFinest, (*sigVals), (*tauAtU), numGpts, gWts,
      bdyArr, LaplacianStencil, GradDivStencil, mu, lambda, alpha, Uin, uTmp);

  //4. Initial Gradient
  PetscReal gNormInit, gNorm;
  computeRHS(damg[nlevels - 1], DAMGGetRHS(damg));
  VecNorm(DAMGGetRHS(damg), NORM_2, &gNormInit);

  if(!rank) {
    std::cout<<"Initial: "<<objValInit<<", "<<gNormInit<<std::endl;
  }

  while(iterCnt < maxIterCnt) {
    iterCnt++;

    //5. Compute Newton step
    if(useSteepestDescent) {
      computeRHS(damg[nlevels - 1], DAMGGetx(damg));
      VecNorm(DAMGGetx(damg), NORM_2, &gNorm);
    } else {
      ot::DAMGSolve(damg);
      PetscReal finalResNorm;
      KSPGetResidualNorm(DAMGGetKSP(damg), &finalResNorm);
      VecNorm(DAMGGetRHS(damg), NORM_2, &gNorm);
      if(!rank) {
        std::cout<<"Final Res Norm: "<<finalResNorm<<
          " Initial Res Norm: "<<gNorm<<std::endl;
      }
      if(finalResNorm > (0.1*gNorm) ) {
        if(!rank) {
          std::cout<<"Using Steepest Descent Step."<<std::endl;
        }
        VecCopy(DAMGGetRHS(damg), DAMGGetx(damg));
      }
    }

    //6. Enforce BC
    enforceBC(daFinest, bdyArr, DAMGGetx(damg));

    //Line Search
    PetscScalar lsFac = 2.0;

    PetscReal stepNorm;
    VecNorm(DAMGGetx(damg), NORM_2, &stepNorm);

    if(gNorm < (fTol*gNormInit)) {
      if(!rank) {
        std::cout<<"GN Exit Type 1"<<std::endl;
      }
      break;
    }

    if(!useSteepestDescent) {
      PetscScalar dirDer;
      VecTDot(DAMGGetx(damg), DAMGGetRHS(damg), &dirDer);
      assert(dirDer > 0.0);
      if(!rank) {
        std::cout<<"Directional Derivative: "<<(-dirDer)<<std::endl;
      }
    }

    do {
      lsFac = 0.5*lsFac;
      //7. Use Uout as tmp vector for line search
      VecWAXPY(Uout, lsFac, DAMGGetx(damg), Uin);

      std::vector<std::vector<double> > tauAtUtmp;

      //8. Interpolate at Tmp Point
      computeTauAtU(daFinest, Ne, (*tau), (*gradTau), Uout, PhiMatStencil,
          numGpts, gPts, tauAtUtmp);

      //9.  Objective at tmp Point
      objVal = evalObjective(daFinest, (*sigVals), tauAtUtmp, numGpts, gWts,
          bdyArr, LaplacianStencil, GradDivStencil, mu, lambda, alpha, Uout, uTmp);

    } while( ((lsFac*stepNorm) >= xTol) && (objVal > objValInit) );

    if((lsFac*stepNorm) < xTol) {
      if(!rank) {
        std::cout<<"GN Exit Type 2"<<std::endl;
        std::cout<<"lsFac: "<<lsFac<<std::endl;
      }
      break;
    }

    //11. Update solution
    VecAXPY(Uin, lsFac, DAMGGetx(damg));

    if( objVal < (fTol*objValInit) ) {
      if(!rank) {
        std::cout<<"GN Exit Type 3"<<std::endl;
      }
      break;
    }
    objValInit = objVal;

    //12. Update Hess Context 
    updateHessContexts(damg, Uin);

    //13. Set New Operators
    if(!useSteepestDescent) {
      ot::DAMGSetKSP(damg, createHessMat,  computeHessMat, computeRHS);
    }

    //14. Display
    if(!rank) {
      std::cout<<iterCnt<<", "<<objVal<<", "<<gNorm<<", "<<(lsFac*stepNorm)<<", "<<lsFac<<std::endl;
    }

  }//end while

  if(!rank) {
    std::cout<<"Final: "<<iterCnt<<", "<<objVal<<", "<<gNorm<<std::endl;
  }

  //Copy solution to output vector
  VecCopy(Uin, Uout);

  if(saveGradient) {
    char fname[256];
    sprintf(fname,"gradJ_%d_%d.dat", rank, npes);
    saveVector(DAMGGetRHS(damg), fname);
  }

}

/*
void computeGradTauAtU_Old(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& gradTauAtU) { 

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  composeXposAtU(dao, U, PhiMatStencil, numGpts, gPts, xPosArr, yPosArr, zPosArr);

  //evaluate at all points
  evalGradFnAtAllPts(xPosArr, yPosArr, zPosArr, Ne, tau, gradTau, gradTauAtU);
}
*/

/*
void computeTauAtU_Old(ot::DA* dao, int Ne, const std::vector<std::vector<double> > & tau,
    const std::vector<std::vector<double> >& gradTau, Vec U, double****** PhiMatStencil,
    int numGpts, double* gPts, std::vector<std::vector<double> >& tauAtU) {

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;

  composeXposAtU(dao, U, PhiMatStencil, numGpts, gPts, xPosArr, yPosArr, zPosArr);

  //evaluate at all points
  evalFnAtAllPts(xPosArr, yPosArr, zPosArr, Ne, tau, gradTau, tauAtU);

}
*/

/*
void computeSigVals_Old(ot::DA* dao, int Ne, const std::vector<std::vector<double> >& sig,
    const std::vector<std::vector<double> >& gradSig, int numGpts,
    double* gPts, std::vector<std::vector<double> >& sigVals) {

  std::vector<double> xPosArr;
  std::vector<double> yPosArr;
  std::vector<double> zPosArr;
  if(dao->iAmActive()) {
    unsigned int maxD = dao->getMaxDepth();
    unsigned int balOctMaxD = maxD - 1;
    double hFac = 1.0/static_cast<double>(1u << balOctMaxD);
    for(dao->init<ot::DA_FLAGS::WRITABLE>();
        dao->curr() < dao->end<ot::DA_FLAGS::WRITABLE>();
        dao->next<ot::DA_FLAGS::WRITABLE>()) {
      Point pt = dao->getCurrentOffset();
      unsigned int idx = dao->curr();
      unsigned int lev = dao->getLevel(idx);
      unsigned int xint = pt.xint();
      unsigned int yint = pt.yint();
      unsigned int zint = pt.zint();
      double hOct = hFac*(static_cast<double>(1u << (maxD - lev)));
      double x0 = hFac*static_cast<double>(xint);
      double y0 = hFac*static_cast<double>(yint);
      double z0 = hFac*static_cast<double>(zint);
      for(int m = 0; m < numGpts; m++) {
        for(int n = 0; n < numGpts; n++) {
          for(int p = 0; p < numGpts; p++) {
            xPosArr.push_back(x0 + (0.5*hOct*(1.0 + gPts[m])));
            yPosArr.push_back(y0 + (0.5*hOct*(1.0 + gPts[n])));
            zPosArr.push_back(z0 + (0.5*hOct*(1.0 + gPts[p])));
          }
        }
      }
    }
  }

  //evaluate at all points
  evalFnAtAllPts(xPosArr, yPosArr, zPosArr, Ne, sig, gradSig, sigVals);

}
*/

