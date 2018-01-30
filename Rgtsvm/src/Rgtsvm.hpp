/*
	Copyright (C) 2017  Zhong Wang

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


/**
	\file Rgtsvm.hpp
	\brief R interface of svm and predict
*/

#ifndef __RGTSVM_HPP__
#define __RGTSVM_HPP__

#define  DEF_BIASED true
#define  DEF_COLUMN_MAJOR true
#define  DEF_SMALL_CLUSTERS false
#define  DEF_ACTIVE_CLUSTERS 64

#define _TRY_EXCEPTIONS_ \
	try {


#define _CATCH_EXCEPTIONS_  \
	}  \
	catch( std::exception& error ) {  \
		g_error = true;  \
		g_errorString = error.what();  \
	}  \
	catch( ... ) {  \
		g_error = true;  \
		g_errorString = "Unknown error";  \
	}

#define _CHECK_EXCEPTIONS_ \
	if(g_error) \
	{ \
		*pnError = 1; \
		Rprintf("Error: %s", g_errorString.c_str()); \
		return; \
	}

#define _CHECK_EXCEPTIONS_RETURN_0_ \
	if(g_error) \
	{ \
		*pnError = 1; \
		Rprintf("Error: %s", g_errorString.c_str()); \
		return(NULL); \
	}

extern "C" void gtsvmtrain_epsregression_C (
		   int	*pSparse,
		   double *pX,
		   int64_t	*pVecOffset,// start from 0
		   int64_t 	*pVecIndex, // start from 1
		   int	*pXrow,
		   int 	*pXcol,
		   int	*pXInnerRow,
		   int 	*pXInnerCol,
		   int	*pXRowIndex,
		   int	*pXColIndex,
		   double *pY,

		   int	*pKernelType,
		   // KernelParameter3 <==> degree in libsvm
		   int	*pDegree,
		   // KernelParameter1 <==> gama in libsvm
		   double *pGamma,
		   // KernelParameter2 <==> coef0 in libsvm
		   double *pCoef0,
		   // pRegularization <==> cost in libsvm
		   double *pCost,
		   double *pTolerance,
		   double *pEpsilon,
		   int	*pShrinking,
		   int	*pFitted,
		   int	*pMaxIter,
		   int	*pNoProgressIgnore,

			//output variables
			//# total iteration
		   int	*npTotalIter,
			//# the total number of support vectors
		   int	*pSV,
			//# the index of support vectors
		   int	*pIndex,
			//# the labels of classes
		   int	*pLabels,
			//# the support vectors of each classes
		   int	*pSVofclass,
			//# - m_bias
		   double *pRho,
			//# dont alpha value for each classes and support vector dim[nr, nclass-1]
		   double *pTrainingAlphas,
		   //# prdict labels for the fitted option
		   double *pPredict,

		   int	*pVerbose,
		   int	*pnError);

extern "C" void gtsvmpredict_epsregression_C  (
		  int	*pDecisionvalues,
		  int	*pProbability,

		  int	*pModelSparse,
		  double *pModelX,
		  int64_t	*pModelVecOffset,
		  int64_t	*pModelVecIndex,
		  int	*pModelRow,
		  int 	 *pModelCol,
		  int	*pModelRowIndex,
		  int	*pModelColIndex,

		  int	*pTotnSV,
		  double *pModelRho,
		  double *pModelAlphas,

		  int	*pKernelType,
		  int	*pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,

		  int	*pSparseX,
		  double *pX,
		  int64_t	*pXVecOffset,
		  int64_t	*pXVecIndex,
		  int 	 *pXrow,
		  int 	 *pXInnerRow,
		  int 	 *pXInnerCol,
		  int	*pXRowIndex,
		  int	*pXColIndex,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError);

extern "C" void gtsvmtrain_classfication_C (
		   int	*pSparse,
		   double *pX,
		   int64_t	*pVecOffset,// start from 0
		   int64_t 	*pVecIndex, // start from 1
		   int	*pXrow,
		   int 	*pXcol,
		   int	*pXInnerRow,
		   int	*pXInnerCol,
		   int	*pXRowIndex,
		   int	*pXColIndex,
		   double *pY,

		   int	*pKernelType,
		   // the number of classes
		   int	*pRclasses,
		   // KernelParameter3 <==> degree in libsvm
		   int	*pDegree,
		   // KernelParameter1 <==> gama in libsvm
		   double *pGamma,
		   // KernelParameter2 <==> coef0 in libsvm
		   double *pCoef0,
		   // pRegularization <==> cost in libsvm
		   double *pCost,
		   double *pClassWeight,
		   double *pTolerance,
		   int	*pFitted,
		   int	*pBiased,
		   int	*pMaxIter,
		   int	*pNoProgressIgnore,

			//output variables
			//# total iteration
		   int	*npTotal_iter,
			//# the total number of classes
		   int	*pClasses,
			//# the total number of support vectors
		   int	*pSV,
			//# the index of support vectors
		   int	*pIndex,
			//# the labels of classes
		   int	*pLabels,
			//# the support vectors of each classes
		   int	*pSVofclass,
			//# dont know
		   double *pRho,
			//# dont alpha value for each classes and support vector dim[nr, nclass-1]
		   double *pTrainingAlphas,
		   double * pPredict,
		   double * pDecision,

		   int	*pVerbose,
		   int	*pnError);

extern "C" void gtsvmpredict_classfication_C  (
		  int	*pDecisionvalues,
		  int	*pProbability,
		  int	*pModelSparse,
		  double *pModelX,
		  int64_t	*pModelVecOffset,
		  int64_t	*pModelVecIndex,
		  int	*pModelRow,
		  int 	 *pModelCol,
		  int	*pModelRowIndex,
		  int 	*pModelColIndex,

		  int	*pNclasses,
		  int	*pTotnSV,
		  double *pModelRho,
		  double *pModelAlphas,

		  int	*pKernelType,
		  int	*pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,

		  int	*pSparseX,
		  double *pX,
		  int64_t	*pXVecOffset,
		  int64_t	*pXVecIndex,
		  int 	 *pXrow,
		  int	*pXInnerRow,
		  int	*pXInnerCol,
		  int	*pXRowIndex,
		  int	*pXColIndex,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError);

extern "C" void gtsvmpredict_classfication_batch  (
		  int	*pDecisionvalues,
		  int	*pProbability,

		  int	*pModelSparse,
		  double *pModelX,
		  int64_t	*pModelVecOffset,
		  int64_t	*pModelVecIndex,
		  int	*pModelRow,
		  int 	 *pModelCol,
		  int	*pModelRowIndex,
		  int	*pModelColIndex,
		  int	*pTotnSV,

		  double *pModelRho,
		  double *pModelAlphas,
		  int	*pNclasses,

		  int	*pKernelType,
		  int	*pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,
		  double *pScaledCenter,
		  double *pScaledScale,

		  char   **pszRDSfile,
		  int	*pLenRDSfile,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError);


extern "C" void gtsvmpredict_epsregression_batch  (
		  int	*pDecisionvalues,
		  int	*pProbability,

		  int	*pModelSparse,
		  double *pModelX,
		  int64_t	*pModelVecOffset,
		  int64_t	*pModelVecIndex,
		  int	*pModelRow,
		  int 	 *pModelCol,
		  int	*pModelRowIndex,
		  int	*pModelColIndex,
		  int	*pTotnSV,

		  double *pModelRho,
		  double *pModelAlphas,

		  int	*pKernelType,
		  int	*pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,
		  double *pScaledCenter,
		  double *pScaledScale,

		  char   **pszRDSfile,
		  int	*pLenRDSfile,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError);


extern "C" void* gtsvmpredict_epsregression_loadsvm  (
		  int	*pModelSparse,
		  double *pModelX,
		  int64_t	*pModelVecOffset,
		  int64_t	*pModelVecIndex,
		  int	*pModelRow,
		  int 	 *pModelCol,
		  int	*pModelRowIndex,
		  int	*pModelColIndex,

		  int	*pTotnSV,
		  double *pModelRho,
		  double *pModelAlphas,

		  int	*pKernelType,
		  int	*pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,

		  int	*pVerbose,
		  int	*pnError);

extern "C" void gtsvmpredict_epsregression_direct_C  (
		  void	 *pModel,
		  int	 *pDecisionvalues,
		  int	 *pProbability,

		  int	 *pSparseX,
		  double *pX,
		  int64_t *pXVecOffset,
		  int64_t *pXVecIndex,
		  int 	 *pXrow,
		  int 	 *pXInnerRow,
		  int 	 *pXInnerCol,
		  int	 *pXRowIndex,
		  int	 *pXColIndex,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError);

extern "C" void* gtsvmpredict_classfication_loadsvm  (
		  int	*pModelSparse,
		  double *pModelX,
		  int64_t	*pModelVecOffset,
		  int64_t	*pModelVecIndex,
		  int	*pModelRow,
		  int 	 *pModelCol,
		  int	*pModelRowIndex,
		  int 	*pModelColIndex,

		  int	*pNclasses,
		  int	*pTotnSV,
		  double *pModelRho,
		  double *pModelAlphas,

		  int	*pKernelType,
		  int	*pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,

		  int	*pVerbose,
		  int	*pnError);

extern "C" void gtsvmpredict_classfication_direct_C  (
		  void  *pModel,
		  int	*pDecisionvalues,
		  int	*pProbability,

		  int	*pSparseX,
		  double *pX,
		  int64_t	*pXVecOffset,
		  int64_t	*pXVecIndex,
		  int 	 *pXrow,
		  int	*pXInnerRow,
		  int	*pXInnerCol,
		  int	*pXRowIndex,
		  int	*pXColIndex,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError);

extern "C" void gtsvmpredict_unloadsvm_C ( void *pModel, int *pnError );

extern "C" int gtsvm_selectDevice( int deviceID, int* nTotal );

extern "C" void gtsvm_resetDevice();


#endif    /* __RGTSVM_HPP__ */
