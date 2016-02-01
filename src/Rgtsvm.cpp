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
		strcpy( *pszError, g_errorString.c_str() ); \
		Rprintf("Error: %s", g_errorString.c_str()); \
		return; \
	}


extern "C" void gtsvmtrain_epsregression (
		   double *pX,
		   int    *pXrow,
		   int 	  *pXcol,
	       double *pY,
	       int    *pVecOffset,// start from 0
	       int 	  *pVecIndex, // start from 1
	       int    *pSparse,

	       int    *pKernelType,
	       // KernelParameter3 <==> degree in libsvm
	       int    *pDegree,
	       // KernelParameter1 <==> gama in libsvm
	       double *pGamma,
	       // KernelParameter2 <==> coef0 in libsvm
	       double *pCoef0,
	       // pRegularization <==> cost in libsvm
	       double *pCost,
	       double *pTolerance,
	       double *pEpsilon,
	       int    *pShrinking,
	       int    *pProbability,
	       int    *pFitted,
	       int    *pMaxIter,
			//output variables
			//# total iteration
	       int    *npTotalIter,
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
			//# dont know
	       double *pSigma,
			//# dont know
	       double *pProbA,
			//# dont know
	       double *pProbB,
	       //# prdict labels for the fitted option
	       double *pPredict,
			//# dont alpha value for each classes and support vector dim[nr, nclass-1]
	       double *pTrainingAlphas,
	       int    *pNoProgressIgnore,
	       int    *pVerbose,
	       char   **pszError)
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

	// GTSVM doesn't have LINEAR function, use the polynomial to replace it.
	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

	Rprintf("pKernel_type=%d sparse=%d[%d,%d] \n", *pKernelType, *pSparse, *pXrow, *pXcol );

	unsigned int nSample = (*pXrow)/2;
	boost::shared_array< float > pLinearTerm( new float[ *pXrow ] );
	boost::shared_array< double > pLabelY( new double[ *pXrow ] );

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
			(void*)pLabelY.get(),
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			columnMajor,
			false,
			regularization,
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

	Rprintf("MaxIter =%d Tolerance=%f Epsilon=%f \n", *pMaxIter, *pTolerance, *pEpsilon );

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

		if(*pVerbose) Rprintf("ii=%d, %f, %f m_bias=%f\n", ii, primal, dual, psvm->GetBias());

		if ( 2 * ( primal - dual ) < (*pTolerance) * ( primal + dual ) )
			break;

		_CATCH_EXCEPTIONS_

		*npTotalIter = ii;

		if(g_error && *pNoProgressIgnore && g_errorString == "An iteration made no progress" )
		{
			Rprintf("Warning: No convergent in the optimization process, ignore.\n");
			strcpy( *pszError, "    ");
			g_error = false;
			break;
		}

		if(g_error) break;
	}

	Rprintf("Iteration = %d \n", *npTotalIter );

	_CHECK_EXCEPTIONS_

	_TRY_EXCEPTIONS_
	boost::shared_array< float > trainingAlphas( new float[ *pXrow ] );
	psvm->GetAlphas( (void*)(trainingAlphas.get()), GTSVM_TYPE_FLOAT, columnMajor );

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

	Rprintf("SV number = %d\n", *pSV );
	Rprintf("rho = %f\n", *pRho );


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
				columnMajor);
		}

		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
		    pPredict[ ii ] = result[ ii ];

		_CATCH_EXCEPTIONS_
		_CHECK_EXCEPTIONS_
	}

	Rprintf("DONE!\n");

}

extern "C" void gtsvmpredict_epsregression  (
		  int    *pDecisionvalues,
		  int    *pProbability,
		  int    *pModelSparse,
		  double *pModelX,
		  int    *pModelRow,
		  int 	 *pModelCol,
		  int    *pModelVecOffset,
		  int    *pModelVecIndex,

		  int    *pTotnSV,
		  double *pModelY,
		  double *pModelRho,
		  double *pModelAlphas,

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
	      int    *pVerbose,
		  char   **pszError)
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

	Rprintf("Model Load (kernel_type=%d Model Sparse=%d[%d,%d])\n", *pKernelType, *pModelSparse, *pModelRow, *pModelCol );

	_TRY_EXCEPTIONS_

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
			(void*)pLabelY.get(),
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			false,
			false,
			regularization,
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
			(void*)pLabelY.get(),
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pModelRow,
			(unsigned int)*pModelCol,
			columnMajor,
			false,
			regularization,
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

	Rprintf("Set Model.\n");

	_TRY_EXCEPTIONS_

	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->SetBias(  -1*(*pModelRho) );
	psvm->ClusterTrainingVectors( smallClusters, activeClusters );


	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	Rprintf("Predicting. (Sparse =%d[%d])\n", *pSparseX, *pXrow);

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
			columnMajor);
	}

	for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
	{
		pDec[ ii ] = result[ ii ];
		pRet[ ii ] = result[ ii ];
	}

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_

	Rprintf("DONE!(Score[%d,%d])\n", *pXrow, *pModelCol );
}

extern "C" void gtsvmtrain_classfication (double *pX,
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
		   double *pTolerance,
	       int    *pFitted,
	       int    *pBiased,
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
	       double * pPredict,
			//# total iteration
	       int    *npTotal_iter,
			//# dont alpha value for each classes and support vector dim[nr, nclass-1]
	       double *pTrainingAlphas,
	       int    *pNoProgressIgnore,
	       int    *pVerbose,
	       char   **pszError)
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

	Rprintf("pKernel_type=%d nclasses=%d biased=%d sparse=%d[%d,%d] \n", *pKernelType, nclasses, *pBiased, *pSparse, *pXrow, *pXcol );

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

	_TRY_EXCEPTIONS_

	boost::shared_array< float > pLinearTerm( new float[ *pXrow ] );
	std::fill( pLinearTerm.get(), pLinearTerm.get() + (*pXrow), 1.0f );

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
			(void*)pY,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
			(unsigned int)*pXrow,
			(unsigned int)*pXcol,
			columnMajor,
			multiclass,
			regularization,
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

	Rprintf("MaxIter =%d Tolerance=%f nClass =%d \n", *pMaxIter, *pTolerance, *pClasses );

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

		if(*pVerbose) Rprintf("ii=%d, %f, %f m_bias=%f\n", ii, primal, dual, psvm->GetBias());

		if ( 2 * ( primal - dual ) < (*pTolerance) * ( primal + dual ) )
			break;

		_CATCH_EXCEPTIONS_

		*npTotal_iter = ii;

		if(g_error && *pNoProgressIgnore && g_errorString == "An iteration made no progress" )
		{
			Rprintf("Warning: No convergent in the optimization process, ignore.\n");
			strcpy( *pszError, "    ");
			g_error = false;
			break;
		}

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

	*pRho = -1.0 * psvm->GetBias();

	Rprintf("SV number = %d\n", *pSV );
	Rprintf("rho = %f\n", *pRho );

	_TRY_EXCEPTIONS_

	psvm->ShrinkClassfication(smallClusters, activeClusters);
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
					if( result[ ii ] < 0)
						pPredict[ ii ]= -1;
					else
						pPredict[ ii ] = 1;
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
					if( result[ ii ] < 0)
						pPredict[ ii ]= -1;
					else
						pPredict[ ii ] = 1;
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

extern "C" void gtsvmpredict_classfication  (
		  int    *pDecisionvalues,
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
		  double *pModelRho,
		  double *pModelAlphas,

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
	      int    *pVerbose,
		  char   **pszError)
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

	Rprintf("Model Load (kernel_type=%d nclasses=%d Sparse=%d[%d,%d])\n", *pKernelType, nclasses, *pModelSparse, *pModelRow, *pModelCol );

	_TRY_EXCEPTIONS_

	boost::shared_array< float > pLinearTerm( new float[ *pXrow ] );
	std::fill( pLinearTerm.get(), pLinearTerm.get() + *pXrow, 1.0f );

    if (*pModelSparse > 0)
		psvm->InitializeSparse(
			(void*)pModelX,
			(size_t*)pModelVecIndex,
			(size_t*)pModelVecOffset,
			GTSVM_TYPE_DOUBLE,
			(void*)pModelY,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
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
			activeClusters,
			false);
    else
		psvm->InitializeDense(
			(void*)pModelX,
			GTSVM_TYPE_DOUBLE,
			(void*)pModelY,
			GTSVM_TYPE_DOUBLE,
			(void*)pLinearTerm.get(),
			GTSVM_TYPE_FLOAT,
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

	unsigned int ncol = psvm->GetClasses();

	Rprintf("Predicting. sparse= %d,  (Dec Matrix =[%d, %d])\n", *pSparseX, *pXrow, ncol);

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
			pRet[ ii ]= n_idx;
		}
	}

	Rprintf("DONE!(Score[%d,%d])\n", *pXrow, ncol );
}
