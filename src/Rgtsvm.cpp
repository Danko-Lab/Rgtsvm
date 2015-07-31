/*
	Copyright (C) 2015  Zhong Wang

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
#include <Rdefines.h>

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
		strcpy( *pszError, g_errorString.c_str() ); \
		Rprintf("Error: %s", g_errorString.c_str()); \
		return; \
	}


extern "C" void gtsvmtrain (double *pX,
		   int    *pXrow,
		   int 	  *pXcol,
	       double *pY,
	       int    *pVecOffset,// start from 0
	       int 	  *pVecIndex, // start from 1
	       int    *pSparse,
	       int    *pKernelType,
		   // the number of classes
	       int    *pRclasses,
	       // KernelParameter3 <==> degree in libsvm
	       int    *pDegree,
	       // KernelParameter1 <==> gama in libsvm
	       double *pGamma,
	       // KernelParameter2 <==> coef0 in libsvm
	       double *pCoef0,
	       // pRegularization <==> cost in libsvm
	       double *pCost,
	       double *pEpsilon,
	       int    *pFitted,
	       int    *pMaxIter,
			//output variables
			//# the total number of classes
	       int    *pClasses,
			//# the total number of support vectors
	       int    *pSV,
			//# the index of support vectors
	       int    *pIndex,
			//# the labels of classes
	       int    *pLabels,
			//# the support vectors of each classes
	       int    *pSVofclass,
			//# dont know
	       double *pRho,
			//# dont alpha value for each classes and support vector dim[nr, nclass-1]
	       double *pTrainingAlphas,
	       double *pTrainingResponses,
	       double *pTrainingNormsSquared,
	       double *pTrainingKernelNormsSquared,

	       double * pPredict,
			//# total iteration
	       int    *npTotal_iter,
	       char   **pszError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	bool biased = false;
	bool columnMajor = true;
	bool smallClusters = false;
	int  activeClusters = 64;
	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	unsigned int nclasses = (unsigned int)(*pRclasses);
	bool multiclass = (nclasses > 2 );

	Rprintf("pKernel_type=%d nclasses=%d sparse=%d[%d,%d] \n", *pKernelType, nclasses, *pSparse, *pXrow, *pXcol );

	// Only 1 class can not do classfication.
	if( nclasses == 1 )
	{
		g_error = true;
		g_errorString = "WARNING: training data in only one class. See README for details.";
	}

	// for multiclass, the values are from 0 to nclass-1. but for binary, the values are 1 and -1
	// the data from R starts from 1
	if( multiclass )
		for(int i=0; i<(*pXrow); i++) pY[i] = pY[i] - 1.0;

	// GTSVM doesn't have LINEAR function, use the polynomial to replace it.
	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	_CHECK_EXCEPTIONS_

	_TRY_EXCEPTIONS_

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
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			false,
			multiclass,
			regularization,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters);
    }
    else
	{
		psvm->InitializeDense(
			(void*)pX,
			GTSVM_TYPE_DOUBLE,
			(void*)pY,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			columnMajor,
			multiclass,
			regularization,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			biased,
			smallClusters,
			activeClusters);
	}

	_CATCH_EXCEPTIONS_

	_CHECK_EXCEPTIONS_

	// for multiclass, m_classes = nclasses, but for binary classfication, m_classes is 1!!!
	*pClasses = (multiclass) ? psvm->GetClasses() : psvm->GetClasses() + 1;

	Rprintf("MaxIter =%d Epsilon=%f nClass =%d \n", *pMaxIter, *pEpsilon, *pClasses );

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

		if ( 2 * ( primal - dual ) < (*pEpsilon) * ( primal + dual ) )
			break;

		_CATCH_EXCEPTIONS_

		*npTotal_iter = ii;

		if(g_error) break;
	}

	_CHECK_EXCEPTIONS_

	Rprintf("Iteration = %d \n", *npTotal_iter );

	//*** for binary classfication, only one Alpha value for each sample.
	unsigned int nCol = psvm->GetClasses();
	boost::shared_array< float > trainingAlphas( new float[ (*pXrow) * nCol ] );
	psvm->GetAlphas( (void*)(trainingAlphas.get()), GTSVM_TYPE_FLOAT, columnMajor );

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

	Rprintf("SV number = %d\n", *pSV );

	_TRY_EXCEPTIONS_

	psvm->Shrink(smallClusters, activeClusters);
	psvm->GetTrainingVectorNormsSquared( (void*)pTrainingNormsSquared, GTSVM_TYPE_DOUBLE );
	psvm->GetTrainingVectorKernelNormsSquared( (void*)pTrainingNormsSquared, GTSVM_TYPE_DOUBLE );
	psvm->GetTrainingResponses( (void*)pTrainingResponses, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->GetAlphas( (void*)pTrainingAlphas, GTSVM_TYPE_DOUBLE, columnMajor );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_


	if(*pFitted)
	{
		_TRY_EXCEPTIONS_

		boost::shared_array< double > result( new double[ (*pXrow) * nCol ] );
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


			if(!multiclass)
			{
				for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
					if( result[ ii ] < 0)  pPredict[ ii ]= -1; else pPredict[ ii ] = 1;
			}
			else
			{
				for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
				{
					unsigned int n_idx = 0;
					for ( unsigned int jj = 1; jj < nCol; ++jj )
					{
						if(	result[ ii*nCol + jj ] > result[ ii*nCol + n_idx ] )
							n_idx = jj;
					}
					pPredict[ii] = (n_idx+1)*1.0;
				}
			}
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
				columnMajor);

			if(!multiclass)
			{
				for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
					if( result[ ii ] < 0)  pPredict[ ii ]= -1; else pPredict[ ii ] = 1;
			}
			else
			{
				for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
				{
					unsigned int n_idx = 0;
					for ( unsigned int jj = 1; jj < nCol; ++jj )
					{
						if(	result[ ii + jj*(*pXrow) ] > result[ ii + n_idx*(*pXrow) ] )
							n_idx = jj;
					}
					pPredict[ii] = (n_idx+1)*1.0;
				}
			}
		}
		_CATCH_EXCEPTIONS_
		_CHECK_EXCEPTIONS_
	}

	Rprintf("DONE!\n");

}

extern "C" void gtsvmpredict  (int    *pDecisionvalues,
		  int    *pProbability,
		  int    *pModelSparse,
		  double *pModelX,
		  int    *pModelRow,
		  int 	 *pModelCol,
		  int    *pModelVecOffset,
		  int    *pModelVecIndex,

		  int    *pNclasses,
		  int    *pTotnSV,
		  double *pModelY,
		  double *pModelAlphas,
		  double *pModelResponses,
		  double *pModelNormsSquared,
		  double *pModelKernelNormsSquared,

		  int    *pKernelType,
		  int    *pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,

		  int    *pSparseX,
		  double *pX,
		  int 	 *pXrow,
		  int    *pXVecOffset,
		  int    *pXVecIndex,

		  double *pRet,
		  double *pDec,
		  double *pProb,
		  char   **pszError)
{
	GTSVM::SVM svm;
	GTSVM::SVM* psvm = &svm;
	bool g_error = false;
	std::string g_errorString;

	bool biased = false;
	bool columnMajor = true;
	bool smallClusters = false;
	int  activeClusters =64;
	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	unsigned int nclasses = (unsigned int)(*pNclasses);
	bool multiclass = (nclasses > 2 );

	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	Rprintf("Model Load (kernel_type=%d nclasses=%d Model Sparse=%d[%d,%d])\n", *pKernelType, nclasses, *pModelSparse, *pModelRow, *pModelCol );

	_TRY_EXCEPTIONS_

    if (*pModelSparse > 0)
		psvm->InitializeSparse(
			(void*)pModelX,
			(size_t*)pModelVecIndex,
			(size_t*)pModelVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)pModelY,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			false,
			multiclass,
			regularization,
			static_cast< GTSVM_Kernel >(*pKernelType),
			kernelParameter1,
			kernelParameter2,
			kernelParameter3,
			biased,
			smallClusters,
			activeClusters);
    else
		psvm->InitializeDense(
			(void*)pModelX,
			GTSVM_TYPE_DOUBLE,
			(void*)pModelY,
			GTSVM_TYPE_DOUBLE,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			columnMajor,
			multiclass,
			regularization,
			static_cast< GTSVM_Kernel >(*pKernelType),
			(float)kernelParameter1,
			(float)kernelParameter2,
			(float)kernelParameter3,
			biased,
			smallClusters,
			activeClusters);

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	Rprintf("Set Model.\n");

	_TRY_EXCEPTIONS_

	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->SetTrainingResponses( (void*)pModelResponses, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->SetTrainingVectorNormsSquared( (void*)pModelNormsSquared, GTSVM_TYPE_DOUBLE );
	psvm->SetTrainingVectorKernelNormsSquared( (void*)pModelKernelNormsSquared, GTSVM_TYPE_DOUBLE );
	psvm->ClusterTrainingVectors( smallClusters, activeClusters );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	Rprintf("Predicting. (Sparse =%d[%d])\n", *pSparseX, *pXrow);

	unsigned int ncol = psvm->GetClasses();
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
			false	);
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
			columnMajor);
		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			for ( unsigned int jj = 0; jj < ncol; ++jj )
				pDec[ ii + jj*(*pXrow) ] = result[ ii + jj*(*pXrow) ];
	}

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_


	if(!multiclass)
	{
		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			if( pDec[ ii ] < 0)  pRet[ ii ]= -1; else pRet[ ii ] = 1;
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
			pRet[ ii ]= n_idx + 1;
		}
	}

	Rprintf("DONE!(Score[%d,%d])\n", *pXrow, ncol );
}
