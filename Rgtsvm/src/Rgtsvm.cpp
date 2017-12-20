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
	\file Rgtsvm.cpp
	\brief R interface of svm and predict
*/

#include "headers.hpp"
#include <R.h>
#include <Rinternals.h>
#include <Rembedded.h>
#include <Rdefines.h>
#include <R_ext/Parse.h>

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


SEXP run_script(char* szCmd)
{
	SEXP cmdSexp, cmdExpr, ans = R_NilValue;
	ParseStatus status;

	PROTECT(cmdSexp = allocVector(STRSXP, 1));
	SET_STRING_ELT(cmdSexp, 0, mkChar(szCmd) );
	cmdExpr = PROTECT(R_ParseVector(cmdSexp, -1, &status, R_NilValue));

	if (status!=PARSE_OK)
	{
		UNPROTECT(2);
		error("invalid call %s", szCmd);
	}

	for(int i=0; i<length(cmdExpr); i++)
		ans = eval(VECTOR_ELT(cmdExpr,i), R_GlobalEnv);

	UNPROTECT(2);
	return(ans);
}


extern "C" void gtsvmtrain_epsregression (
		   int	*pSparse,
		   double *pX,
		   int	*pVecOffset,// start from 0
		   int 	*pVecIndex, // start from 1
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
		   int	*pnError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	*pnError = 0;

	bool biased = DEF_BIASED;
	bool columnMajor = DEF_COLUMN_MAJOR;
	bool smallClusters = DEF_SMALL_CLUSTERS;
	int  activeClusters = DEF_ACTIVE_CLUSTERS;

	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;

	// GTSVM doesn't have LINEAR function, use the polynomial to replace it.
	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	char szSparse=' ';
	if(*pSparse) szSparse='S';

	unsigned int nSample = (*pXrow)/2;
	if(*pVerbose) Rprintf("[e-SVR training] X = [%d,%d%c] kernel = %d degree = %f gamma = %f, c0 = %f C = %f\n",
			nSample, *pXcol, szSparse, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization );

	boost::shared_array< float > pLinearTerm( new float[ *pXrow ] );
	boost::shared_array< double > pLabelY( new double[ *pXrow ] );

	// c(-1,0,1) + 1 ==> c(0, 1,2)
	boost::shared_array< float > regularizationWeights( new float[ 3 ] );
	std::fill( regularizationWeights.get(), regularizationWeights.get() + 3, 1.0f );

	_TRY_EXCEPTIONS_

	for(unsigned int ii=0; ii<nSample; ii++)
	{
		pLabelY[ ii ] = 1.0f;
		pLabelY[ ii + nSample] = -1.0f;
		pLinearTerm[ ii ] = pY[ii] - (*pEpsilon);
		pLinearTerm[ ii + nSample] = - pY[ii] - (*pEpsilon);
	}

	if (*pSparse > 0)
	{
		psvm->InitializeSparse(
			(void*)pX,
			// sizeof (size_t)==8 <==> as.integer64(bit64 package)
			(size_t*)pVecIndex,
			(size_t*)pVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)pLabelY.get(),
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			false,
			false,
			regularization,
			regularizationWeights.get(),
			2,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			true);
	}
	else
	{
		psvm->InitializeDense(
			(void*)pX,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			(unsigned int)*pXInnerRow,
			(unsigned int)*pXInnerCol,
			(unsigned int*)pXRowIndex,
			(unsigned int*)pXColIndex,
			(void*)pLabelY.get(),
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			columnMajor,
			false,
			regularization,
			regularizationWeights.get(),
			2,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			true);
	}

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pVerbose) Rprintf("[e-SVR training] MaxIter=%d tolerance=%f epsilon=%f \n", *pMaxIter, *pTolerance, *pEpsilon );

	// must be a multiple of 16
	unsigned int const repetitions = 256;
	for ( unsigned int ii = 0; ii < (unsigned int)(*pMaxIter); ii += repetitions )
	{
		double primal =  std::numeric_limits< double >::infinity();
		double dual   = -std::numeric_limits< double >::infinity();

		_TRY_EXCEPTIONS_

		std::pair< CUDA_FLOAT_DOUBLE, CUDA_FLOAT_DOUBLE > const result = psvm->Optimize( repetitions );
		primal = result.first;
		dual   = result.second;

		//for debug
		//if(*pVerbose) Rprintf("ii=%d, %f, %f m_bias=%f\n", ii, primal, dual, psvm->GetBias());

		if ( 2 * ( primal - dual ) < (*pTolerance) * ( primal + dual ) )
			break;

		_CATCH_EXCEPTIONS_

		*npTotalIter = ii;

		if(g_error && *pNoProgressIgnore && g_errorString == "An iteration made no progress" )
		{
			Rprintf("Warning: No convergent in the optimization process, ignore.\n");
			*pnError = 0;
			g_error = false;
			break;
		}

		if(g_error) break;
	}

	_CHECK_EXCEPTIONS_

	_TRY_EXCEPTIONS_
	boost::shared_array< double > trainingAlphas( new double[ *pXrow ] );
	psvm->GetAlphas( (void*)(trainingAlphas.get()), GTSVM_TYPE_DOUBLE, columnMajor );

	*pSV = 0;
	int nLableFill = 0;
	for ( unsigned int ii = 0; ii < nSample; ++ii ) {

		if ( trainingAlphas[ii] - trainingAlphas[ii + nSample ] != 0.0 )
		{
			*pIndex = ii + 1;
			pIndex ++;
			*pSV = (*pSV) + 1;

			bool bFound=false;
			for(int k=0; k<nLableFill;k++)
			{
				if(pLabels[k] == (int)(pLabelY[ii]) )
				{
					pSVofclass[k] = pSVofclass[k] + 1;
					bFound=true;
				}
			}

			if(!bFound)
			{
				pLabels[nLableFill] = (int)(pLabelY[ii]);
				pSVofclass[nLableFill] = 1;
				nLableFill++;
			}
		}
	}


	*pRho = -1.0 * psvm->GetBias();

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pVerbose) Rprintf("[e-SVR training] Iteration=%d SV.number=%d rho=%f\n", *npTotalIter, *pSV, *pRho );

	_TRY_EXCEPTIONS_

	psvm->ShrinkRegression( smallClusters, activeClusters );
	psvm->GetAlphas( (void*)pTrainingAlphas, GTSVM_TYPE_DOUBLE, columnMajor );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pFitted)
	{
		_TRY_EXCEPTIONS_

		boost::shared_array< double > result( new double[ *pXrow] );
		if (*pSparse > 0)
		{
			psvm->ClassifySparse(
				(void*)(result.get()),
				GTSVM_TYPE_DOUBLE,
				(void*)pX,
				(size_t*)pVecIndex,
				(size_t*)pVecOffset,
				GTSVM_TYPE_DOUBLE,
				(unsigned)*pXrow,
				(unsigned)*pXcol,
				false	);
		}
		else
		{
			psvm->ClassifyDense(
				(void*)(result.get()),
				GTSVM_TYPE_DOUBLE,
				(void*)pX,
				GTSVM_TYPE_DOUBLE,
				(unsigned)*pXrow,
				(unsigned)*pXcol,
				(unsigned int)*pXInnerRow,
				(unsigned int)*pXInnerCol,
				(unsigned int*)pXRowIndex,
				(unsigned int*)pXColIndex,
				columnMajor);
		}

		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			pPredict[ ii ] = result[ ii ];

		_CATCH_EXCEPTIONS_
		_CHECK_EXCEPTIONS_
	}

	if(*pVerbose) Rprintf("[e-SVR training] DONE!\n");

}

extern "C" void gtsvmpredict_epsregression_batch  (
		  int	*pDecisionvalues,
		  int	*pProbability,

		  int	*pModelSparse,
		  double *pModelX,
		  int	*pModelVecOffset,
		  int	*pModelVecIndex,
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
		  int	*pnError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	bool biased = DEF_BIASED;
	bool columnMajor = DEF_COLUMN_MAJOR;
	bool smallClusters = DEF_SMALL_CLUSTERS;
	int  activeClusters = DEF_ACTIVE_CLUSTERS;

	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	*pnError = 0;

	if (*pVerbose) Rprintf("[e-SVR batch#] X=[?,%d] kernel=%d degree=%f gamma=%f, c0=%f C=%f\n",
			 *pModelCol, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization );

	_TRY_EXCEPTIONS_

	// c(-1,0,1) + 1 ==> c(0, 1,2)
	boost::shared_array< float > regularizationWeights( new float[ 3 ] );
	std::fill( regularizationWeights.get(), regularizationWeights.get() + 3, 1.0f );

	boost::shared_array< double > pLabelY( new double[ *pModelRow ] );
	boost::shared_array< float > pLinearTerm( new float[ *pModelRow ] );
	for(int i=0;i<(*pModelRow); i++)
	{
		pLabelY[ i ] = 1.0;
		pLinearTerm[ i ] = 1.0;
	}

	if (*pModelSparse > 0)
		psvm->InitializeSparse(
			(void*)pModelX,
			(size_t*)pModelVecIndex,
			(size_t*)pModelVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			false,
			false,
			regularization,
			regularizationWeights.get(),
			2,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);
	else
		psvm->InitializeDense(
			(void*)pModelX,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int*)pModelRowIndex,
			(unsigned int*)pModelColIndex,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			columnMajor,
			false,
			regularization,
			regularizationWeights.get(),
			2,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);


	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pVerbose) Rprintf("[e-SVR batch#] Model=%d[%d,%d] rho=%f\n", *pModelSparse, *pModelRow, *pModelCol, *pModelRho );

	_TRY_EXCEPTIONS_

	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->SetBias(  -1*(*pModelRho) );
	psvm->ClusterTrainingVectors( smallClusters, activeClusters );


	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	int nOffsetret = 0;
	for (int k=0; k< *pLenRDSfile; k++)
	{
		_TRY_EXCEPTIONS_

		char szCmd[1024]={0};
		sprintf( szCmd, "readRDS('%s')", pszRDSfile[k] );

		if(*pVerbose) Rprintf("			   Loading %s\n", pszRDSfile[k] );

		SEXP pNewData = run_script( szCmd );

		SEXP dims = getAttrib(pNewData, R_DimSymbol);
		int nRow=0, nCol=0;

		if (length(dims) == 2)
		{
			nRow = INTEGER(dims)[0];
			nCol = INTEGER(dims)[1];
		}
		else if(length(dims) == 1)
		{
			nRow = 1;
			nCol = INTEGER(dims)[0];
		}
		else
			return;

		boost::shared_array< double > result( new double[ nRow] );
		boost::shared_array< unsigned int > pXRowIndex( new unsigned int[ nRow ] );
		boost::shared_array< unsigned int > pXColIndex( new unsigned int[ nCol ] );

		for(int i=0; i<nRow; i++) pXRowIndex[i] = i;
		for(int i=0; i<nCol; i++) pXColIndex[i] = i;

		psvm->ClassifyDense(
				(void*)result.get(),
				GTSVM_TYPE_DOUBLE,
				(void*)REAL( pNewData ),
				GTSVM_TYPE_DOUBLE,
				(unsigned)nRow,
				(unsigned)*pModelCol,
				(unsigned int)nRow,
				(unsigned int)nCol,
				(unsigned int*)pXRowIndex.get(),
				(unsigned int*)pXColIndex.get(),
				columnMajor);

		for ( unsigned int ii = 0; ii < (unsigned int)nRow; ++ii )
		{
			pDec[ ii + nOffsetret] = result[ ii ];
			pRet[ ii + nOffsetret] = result[ ii ];
		}

		nOffsetret = nOffsetret + nRow;

		_CATCH_EXCEPTIONS_
		_CHECK_EXCEPTIONS_
	}

	if(*pVerbose) Rprintf("[e-SVR batch#] DONE!\n");
}


extern "C" void gtsvmpredict_epsregression  (
		  int	*pDecisionvalues,
		  int	*pProbability,

		  int	*pModelSparse,
		  double *pModelX,
		  int	*pModelVecOffset,
		  int	*pModelVecIndex,
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
		  int	*pXVecOffset,
		  int	*pXVecIndex,
		  int 	 *pXrow,
		  int 	 *pXInnerRow,
		  int 	 *pXInnerCol,
		  int	*pXRowIndex,
		  int	*pXColIndex,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	bool biased = DEF_BIASED;
	bool columnMajor = DEF_COLUMN_MAJOR;
	bool smallClusters = DEF_SMALL_CLUSTERS;
	int  activeClusters = DEF_ACTIVE_CLUSTERS;

	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	*pnError = 0;

	if(*pVerbose) Rprintf("[e-SVR predict#] X=%d[%d,%d] kernel=%d degree=%f gamma=%f, c0=%f C=%f\n",
			*pSparseX, *pXrow, *pModelCol, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization );

	_TRY_EXCEPTIONS_

	// c(-1,0,1) + 1 ==> c(0, 1,2)
	boost::shared_array< float > regularizationWeights( new float[ 3 ] );
	std::fill( regularizationWeights.get(), regularizationWeights.get() + 3, 1.0f );

	boost::shared_array< double > pLabelY( new double[ *pModelRow ] );
	boost::shared_array< float > pLinearTerm( new float[ *pModelRow ] );
	for(int i=0;i<(*pModelRow); i++)
	{
		pLabelY[ i ] = 1.0;
		pLinearTerm[ i ] = 1.0;
	}

	if (*pModelSparse > 0)
		psvm->InitializeSparse(
			(void*)pModelX,
			(size_t*)pModelVecIndex,
			(size_t*)pModelVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			false,
			false,
			regularization,
			regularizationWeights.get(),
			2,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);
	else
		psvm->InitializeDense(
			(void*)pModelX,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int*)pModelRowIndex,
			(unsigned int*)pModelColIndex,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			columnMajor,
			false,
			regularization,
			regularizationWeights.get(),
			2,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);


	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pVerbose) Rprintf("[e-SVR predict#] Model=%d[%d,%d] rho=%f\n", *pModelSparse, *pModelRow, *pModelCol, *pModelRho );

	_TRY_EXCEPTIONS_

	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->SetBias(  -1*(*pModelRho) );
	psvm->ClusterTrainingVectors( smallClusters, activeClusters );


	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	boost::shared_array< double > result( new double[ (*pXrow)] );

	_TRY_EXCEPTIONS_

	if (*pSparseX > 0)
	{
		psvm->ClassifySparse(
			(void*)(result.get()),
			GTSVM_TYPE_DOUBLE,
			(void*)pX,
			(size_t*)pXVecIndex,
			(size_t*)pXVecOffset,
			GTSVM_TYPE_DOUBLE,
			(unsigned)*pXrow,
			(unsigned)*pModelCol,
			false	);
	}
	else
	{
		psvm->ClassifyDense(
			(void*)(result.get()),
			GTSVM_TYPE_DOUBLE,
			(void*)pX,
			GTSVM_TYPE_DOUBLE,
			(unsigned)*pXrow,
			(unsigned)*pModelCol,
			(unsigned int)*pXInnerRow,
			(unsigned int)*pXInnerCol,
		  	(unsigned int*)pXRowIndex,
		  	(unsigned int*)pXColIndex,
			columnMajor);
	}

	for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
	{
		pDec[ ii ] = result[ ii ];
		pRet[ ii ] = result[ ii ];
	}

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pVerbose) Rprintf("[e-SVR predict#] DONE!\n");
}

extern "C" void gtsvmtrain_classfication (
		   int	*pSparse,
		   double *pX,
		   int	*pVecOffset,// start from 0
		   int 	*pVecIndex, // start from 1
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
		   int	*pnError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	bool columnMajor = DEF_COLUMN_MAJOR;
	bool smallClusters = DEF_SMALL_CLUSTERS;
	int  activeClusters = DEF_ACTIVE_CLUSTERS;
	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	unsigned int nclasses = (unsigned int)(*pRclasses);
	bool multiclass = (nclasses > 2 );

	*pnError = 0;

	// Only 1 class can not do classfication.
	if( nclasses == 1 )
	{
		g_error = true;
		g_errorString = "WARNING: training data in only one class. See README for details.";
	}

	// GTSVM doesn't have LINEAR function, use the polynomial to replace it.
	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	if(*pVerbose) Rprintf("[C-SVC training] X=%d[%d,%d] nclasses=%d biased=%d kernel=%d degree=%f gamma=%f, c0=%f C=%f tolerance=%f\n",
		*pSparse, *pXrow, *pXcol, nclasses, *pBiased, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization, *pTolerance );

	_TRY_EXCEPTIONS_

	boost::shared_array< float > pLinearTerm( new float[ *pXrow ] );
	std::fill( pLinearTerm.get(), pLinearTerm.get() + (*pXrow), 1.0f );

	// c(-1,0,1) + 1 ==> c(0, 1,2)
	boost::shared_array< float > regularizationWeights( new float[ nclasses + 1 ] );
	std::fill( regularizationWeights.get(), regularizationWeights.get() + nclasses + 1, 1.0f );
	if( pClassWeight != NULL )
	{
		if( nclasses==2 )
		{
			regularizationWeights[0] = pClassWeight[0];
			regularizationWeights[2] = pClassWeight[1];
		}
		else
		{
			for(unsigned int ii=0; ii<nclasses; ii++)
				regularizationWeights[ii] = pClassWeight[ii];
		}
	}

	if (*pSparse > 0)
	{
		psvm->InitializeSparse(
			(void*)pX,
			// sizeof (size_t)==8 <==> as.integer64(bit64 package)
			(size_t*)pVecIndex,
			(size_t*)pVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)pY,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			false,
			multiclass,
			regularization,
			regularizationWeights.get(),
			nclasses,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			*pBiased,
			smallClusters,
			activeClusters,
			true);
	}
	else
	{
		psvm->InitializeDense(
			(void*)pX,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			(unsigned int)*pXInnerRow,
			(unsigned int)*pXInnerCol,
			(unsigned int*)pXRowIndex,
			(unsigned int*)pXColIndex,
			(void*)pY,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			columnMajor,
			multiclass,
			regularization,
			regularizationWeights.get(),
			nclasses,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			*pBiased,
			smallClusters,
			activeClusters,
			true);
	}

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	// for multiclass, m_classes = nclasses, but for binary classfication, m_classes is 1!!!
	*pClasses = (multiclass) ? psvm->GetClasses() : psvm->GetClasses() + 1;

	if(*pVerbose) Rprintf("[C-SVC training] MaxIter=%d tolerance=%f class=%d \n", *pMaxIter, *pTolerance, *pClasses );

	// must be a multiple of 16
	unsigned int const repetitions = 256;
	for ( unsigned int ii = 0; ii < (unsigned int)(*pMaxIter); ii += repetitions )
	{
		double primal =  std::numeric_limits< double >::infinity();
		double dual   = -std::numeric_limits< double >::infinity();

		_TRY_EXCEPTIONS_

		std::pair< CUDA_FLOAT_DOUBLE, CUDA_FLOAT_DOUBLE > const result = psvm->Optimize( repetitions );
		primal = result.first;
		dual   = result.second;

		// for debug
		//if(*pVerbose) Rprintf("ii=%d, %f, %f m_bias=%f\n", ii, primal, dual, psvm->GetBias());

		if ( 2 * ( primal - dual ) < (*pTolerance) * ( primal + dual ) )
			break;

		_CATCH_EXCEPTIONS_

		*npTotal_iter = ii;

		if(g_error && *pNoProgressIgnore && g_errorString == "An iteration made no progress" )
		{
			Rprintf("Warning: No convergent in the optimization process, ignore.\n");
			*pnError = 0;
			g_error = false;
			break;
		}

		if(g_error) break;
	}

	_CHECK_EXCEPTIONS_

	//*** for binary classfication, only one Alpha value for each sample.
	unsigned int nCol = psvm->GetClasses();
	boost::shared_array< double > trainingAlphas( new double[ (*pXrow) * nCol ] );
	psvm->GetAlphas( (void*)(trainingAlphas.get()), GTSVM_TYPE_DOUBLE, columnMajor );

	*pSV = 0;
	int nLableFill = 0;
	for ( unsigned int ii = 0; ii < (unsigned int)(*pXrow); ++ii ) {

		bool zero = true;
		for ( unsigned int jj = 0; jj < nCol; ++jj ) {

			if ( trainingAlphas[ jj * (*pXrow) + ii ] != 0 ) {
				zero = false;
				break;
			}
		}


		if ( ! zero )
		{
			*pIndex = ii + 1;
			pIndex ++;
			*pSV = (*pSV) + 1;

			bool bFound=false;
			for(int k=0; k<nLableFill;k++)
			{
				if(pLabels[k] == (int)(pY[ii]) )
				{
					pSVofclass[k] = pSVofclass[k] + 1;
					bFound=true;
				}
			}

			if(!bFound)
			{
				pLabels[nLableFill] = (int)(pY[ii]);
				pSVofclass[nLableFill] = 1;
				nLableFill++;
			}
		}
	}

	*pRho = -1.0 * psvm->GetBias();

	if(*pVerbose) Rprintf("[C-SVC training] Iteration=%d SV.number=%d rho=%f\n", *npTotal_iter, *pSV, *pRho );

	_TRY_EXCEPTIONS_

	psvm->ShrinkClassfication(smallClusters, activeClusters);
	psvm->GetAlphas( (void*)pTrainingAlphas, GTSVM_TYPE_DOUBLE, columnMajor );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pFitted)
	{
		_TRY_EXCEPTIONS_

		boost::shared_array< double > result( new double[ (*pXrow) * nCol ] );
		bool resultColumnMajor = false;

		if (*pSparse > 0)
		{
			psvm->ClassifySparse(
				(void*)(result.get()),
				GTSVM_TYPE_DOUBLE,
				(void*)pX,
				(size_t*)pVecIndex,
				(size_t*)pVecOffset,
				GTSVM_TYPE_DOUBLE,
				(unsigned)*pXrow,
				(unsigned)*pXcol,
				false);
		}
		else
		{
			psvm->ClassifyDense(
				(void*)(result.get()),
				GTSVM_TYPE_DOUBLE,
				(void*)pX,
				GTSVM_TYPE_DOUBLE,
				(unsigned)*pXrow,
				(unsigned)*pXcol,
				(unsigned int)*pXInnerRow,
				(unsigned int)*pXInnerCol,
				(unsigned int*)pXRowIndex,
				(unsigned int*)pXColIndex,
				columnMajor);

			resultColumnMajor = columnMajor;
		}

		if(!multiclass)
		{
			for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			{
				pDecision[ ii ] = result[ ii ];
				if( result[ ii ] < 0)
					pPredict[ ii ]= -1;
				else
					pPredict[ ii ] = 1;
			}
		}
		else
		{

			for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			{
				unsigned int n_idx = 0;

				if(resultColumnMajor)
				{
					pDecision[ ii ] = result[ ii ];
					for ( unsigned int jj = 1; jj < nCol; ++jj )
					{
						pDecision[ ii + jj*(*pXrow) ] = result[ ii + jj*(*pXrow) ];
						if(	result[ ii + jj*(*pXrow) ] > result[ ii + n_idx*(*pXrow) ] )
							n_idx = jj;
					}
				}
				else
				{
					pDecision[ ii ] = result[ ii*nCol + 0 ];
					for ( unsigned int jj = 1; jj < nCol; ++jj )
					{
						pDecision[ ii + jj*(*pXrow) ] = result[ ii*nCol + jj ];
						if(	result[ ii*nCol + jj ] > result[ ii*nCol + n_idx ] )
							n_idx = jj;
					}
				}

				pPredict[ii] = (n_idx+1)*1.0;
			}
		}

		_CATCH_EXCEPTIONS_
		_CHECK_EXCEPTIONS_
	}


	if(*pVerbose) Rprintf("[C-SVC training] DONE!\n");

}

extern "C" void gtsvmpredict_classfication  (
		  int	*pDecisionvalues,
		  int	*pProbability,
		  int	*pModelSparse,
		  double *pModelX,
		  int	*pModelVecOffset,
		  int	*pModelVecIndex,
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
		  int	*pXVecOffset,
		  int	*pXVecIndex,
		  int 	 *pXrow,
		  int	*pXInnerRow,
		  int	*pXInnerCol,
		  int	*pXRowIndex,
		  int	*pXColIndex,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  int	*pVerbose,
		  int	*pnError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	bool biased = DEF_BIASED;
	bool columnMajor = DEF_COLUMN_MAJOR;
	bool smallClusters = DEF_SMALL_CLUSTERS;
	int  activeClusters = DEF_ACTIVE_CLUSTERS;

	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	unsigned int nclasses = (unsigned int)(*pNclasses);
	bool multiclass = (nclasses > 2 );
	if (multiclass) biased = FALSE;

	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	*pnError = 0;

	if(*pVerbose) Rprintf("[C-SVC predict*] X=%d[%d,%d] nclasses=%d, kernel=%d degree=%f gamma=%f, c0=%f C=%f\n",
			*pSparseX, *pXrow, *pModelCol, nclasses, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization );

	_TRY_EXCEPTIONS_

	boost::shared_array< float > pLinearTerm( new float[ *pModelRow ] );
	std::fill( pLinearTerm.get(), pLinearTerm.get() + *pModelRow, 1.0f );

	boost::shared_array< float > regularizationWeights( new float[ *pNclasses + 1 ] );
	std::fill( regularizationWeights.get(), regularizationWeights.get() + *pNclasses + 1, 1.0f );

	if (*pModelSparse > 0)
		psvm->InitializeSparse(
			(void*)pModelX,
			(size_t*)pModelVecIndex,
			(size_t*)pModelVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			false,
			multiclass,
			regularization,
			regularizationWeights.get(),
			nclasses,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);
	else
		psvm->InitializeDense(
			(void*)pModelX,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int*)pModelRowIndex,
			(unsigned int*)pModelColIndex,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			columnMajor,
			multiclass,
			regularization,
			regularizationWeights.get(),
			nclasses,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	_TRY_EXCEPTIONS_

	//psvm->ClusterTrainingVectors( smallClusters, activeClusters );
	psvm->SetBias(  -1*(*pModelRho) );
	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	unsigned int ncol = psvm->GetClasses();

	if(*pVerbose) Rprintf("[C-SVC predict*] Model=%d[%d,%d] rho=%f\n", *pModelSparse, *pModelRow, *pModelCol, *pModelRho );

	boost::shared_array< double > result( new double[ (*pXrow) * ncol ] );

	_TRY_EXCEPTIONS_

	if (*pSparseX > 0)
	{
		psvm->ClassifySparse(
			(void*)(result.get()),
			GTSVM_TYPE_DOUBLE,
			(void*)pX,
			(size_t*)pXVecIndex,
			(size_t*)pXVecOffset,
			GTSVM_TYPE_DOUBLE,
			(unsigned)*pXrow,
			(unsigned)*pModelCol,
			false );

		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			for ( unsigned int jj = 0; jj < ncol; ++jj )
				pDec[ ii + jj*(*pXrow) ] = result[ ii*ncol + jj ];
	}
	else
	{
		psvm->ClassifyDense(
			(void*)(result.get()),
			GTSVM_TYPE_DOUBLE,
			(void*)pX,
			GTSVM_TYPE_DOUBLE,
			(unsigned)*pXrow,
			(unsigned)*pModelCol,
			(unsigned int)*pXInnerRow,
			(unsigned int)*pXInnerCol,
			(unsigned int*)pXRowIndex,
			(unsigned int*)pXColIndex,
			columnMajor );

		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			for ( unsigned int jj = 0; jj < ncol; ++jj )
				pDec[ ii + jj*(*pXrow) ] = result[ii + jj*(*pXrow) ];
	}

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(!multiclass)
	{
		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			if( pDec[ ii ] < 0)
				pRet[ ii ]= -1;
			else
				pRet[ ii ] = 1;
	}
	else
	{
		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
		{
			unsigned int n_idx = 0;
			for ( unsigned int jj = 1; jj < ncol; ++jj )
			{
				if(	pDec[ ii + jj*(*pXrow) ] > pDec[ ii + n_idx*(*pXrow) ] )
					n_idx = jj;
			}
			pRet[ ii ]= (n_idx+1)*1.0;
		}
	}

	if(*pVerbose) Rprintf("[C-SVC predict*] DONE!\n");
}


extern "C" void gtsvmpredict_classfication_batch  (
		  int	*pDecisionvalues,
		  int	*pProbability,

		  int	*pModelSparse,
		  double *pModelX,
		  int	*pModelVecOffset,
		  int	*pModelVecIndex,
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
		  int	*pnError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	bool biased = DEF_BIASED;
	bool columnMajor = DEF_COLUMN_MAJOR;
	bool smallClusters = DEF_SMALL_CLUSTERS;
	int  activeClusters = DEF_ACTIVE_CLUSTERS;

	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	unsigned int nclasses = (unsigned int)(*pNclasses);

	bool multiclass = (nclasses > 2 );
	if (multiclass) biased = FALSE;

	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	*pnError = 0;

	if(*pVerbose) Rprintf("[C-SVC batch*] X=[?,%d] nclasses=%d, kernel=%d degree=%f gamma=%f, c0=%f C=%f\n",
			*pModelCol, nclasses, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization );

	_TRY_EXCEPTIONS_

	boost::shared_array< float > pLinearTerm( new float[ *pModelRow ] );
	std::fill( pLinearTerm.get(), pLinearTerm.get() + *pModelRow, 1.0f );

	boost::shared_array< float > regularizationWeights( new float[ *pNclasses + 1 ] );
	std::fill( regularizationWeights.get(), regularizationWeights.get() + *pNclasses + 1, 1.0f );

	if (*pModelSparse > 0)
		psvm->InitializeSparse(
			(void*)pModelX,
			(size_t*)pModelVecIndex,
			(size_t*)pModelVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			false,
			multiclass,
			regularization,
			regularizationWeights.get(),
			*pNclasses,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);
	else
		psvm->InitializeDense(
			(void*)pModelX,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			(unsigned int*)pModelRowIndex,
			(unsigned int*)pModelColIndex,
			(void*)NULL,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			columnMajor,
			multiclass,
			regularization,
			regularizationWeights.get(),
			*pNclasses,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			biased,
			smallClusters,
			activeClusters,
			false);

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	_TRY_EXCEPTIONS_

	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->SetBias(  -1*(*pModelRho) );
	psvm->ClusterTrainingVectors( smallClusters, activeClusters );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	if(*pVerbose) Rprintf("[C-SVC batch*] Model=%d[%d,%d] rho=%f\n", *pModelSparse, *pModelRow, *pModelCol, *pModelRho );

	unsigned int nclass = psvm->GetClasses();

	char szCmd[1024]={0};
	unsigned int nRow=0, nCol=0,nOffsetret = 0;

	for (int k=0; k< *pLenRDSfile; k++)
	{
		_TRY_EXCEPTIONS_

		// load RDS file and get row count and column count.
		sprintf( szCmd, "readRDS('%s')", pszRDSfile[k] );
		if(*pVerbose) Rprintf("	       Loading %s\n", pszRDSfile[k] );

		SEXP pNewData = run_script( szCmd );
		SEXP dims = getAttrib(pNewData, R_DimSymbol);
		if (length(dims) == 2)
		{
			nRow = INTEGER(dims)[0];
			nCol = INTEGER(dims)[1];
		}
		else if(length(dims) == 1)
		{
			nRow = 1;
			nCol = INTEGER(dims)[0];
		}
		else
			return;

		boost::shared_array< double > result( new double[ nRow * nclass] );
		boost::shared_array< unsigned int > pXRowIndex( new unsigned int[ nRow ] );
		boost::shared_array< unsigned int > pXColIndex( new unsigned int[ nCol ] );
		for(unsigned int i=0; i<nRow; i++) pXRowIndex[i] = i;
		for(unsigned int i=0; i<nCol; i++) pXColIndex[i] = i;

		psvm->ClassifyDense(
			(void*)(result.get()),
			GTSVM_TYPE_DOUBLE,
			(void*)REAL( pNewData ),
			GTSVM_TYPE_DOUBLE,
			(unsigned)nRow,
			(unsigned)*pModelCol,
			(unsigned int)nRow,
			(unsigned int)nCol,
			(unsigned int*)pXRowIndex.get(),
			(unsigned int*)pXColIndex.get(),
			false);

		for ( unsigned int ii = 0; ii < nRow; ++ii )
			for ( unsigned int jj = 0; jj < nclass; ++jj )
				pDec[ (ii + nOffsetret)*nclass + jj  ] = result[ ii*nclass + jj ];

		_CATCH_EXCEPTIONS_
		_CHECK_EXCEPTIONS_

		nOffsetret = nOffsetret + nRow;
	}

	if(!multiclass)
	{
		for ( unsigned int ii = 0; ii < nOffsetret; ++ii )
			if( pDec[ ii ] < 0)
				pRet[ ii ]= -1;
			else
				pRet[ ii ] = 1;
	}
	else
	{
		for (unsigned int ii = 0; ii < nOffsetret; ++ii )
		{
			unsigned int n_idx = 0;
			for ( unsigned int jj = 1; jj < nclass; ++jj )
			{
				if(	pDec[ ii*nclass + jj ] > pDec[ ii*nclass + n_idx ] )
					n_idx = jj;
			}
			pRet[ ii ]= (n_idx+1)*1.0;
		}

		boost::shared_array< double > pDec2( new double[ nOffsetret * nclass] );
		for(unsigned int ii = 0; ii < nOffsetret * nclass; ++ii )
			pDec2[ii] = pDec[ii];

		for(unsigned int ii = 0; ii < nOffsetret; ++ii )
			for ( unsigned int jj = 0; jj < nclass; ++jj )
				pDec[ ii + jj*nOffsetret ] = pDec2[ ii*nclass + jj ];

	}

	if(*pVerbose) Rprintf("[C-SVC batch#] DONE!\n");
}

