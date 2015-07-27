/*
	Copyright (C) 2011  Andrew Cotter

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
	\file gtsvm.cpp
	\brief implementation of C interface to SVM class
*/

#include "headers.hpp"
#include <R.h>
#include <Rdefines.h>

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
	       double *pTolerance,
	       int    *pShrinking,
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
	int  activeClusters =64;
	float regularization = *pCost;
	float kernelParameter1 = *pGamma;
	float kernelParameter2 = *pCoef0;
	float kernelParameter3 = *pDegree;
	unsigned int nclasses = (unsigned int)(*pRclasses);
	bool multiclass = (nclasses > 2 );

	if( nclasses == 1 )
	{
		g_errorString="WARNING: training data in only one class. See README for details.";
		strcpy( *pszError, g_errorString.c_str() );
		Rprintf("Error: %s", g_errorString.c_str());
		return;
	}

	Rprintf("*pKernel_type=%d *nclasses=%d *pSparse=%d[%d,%d] First=%f, Last=%f\n", *pKernelType, nclasses, *pSparse, *pXrow, *pXcol, pY[0], pY[ *pXrow - 1] );

	if ( *pKernelType == 0 )
	{
		*pKernelType = GTSVM_KERNEL_POLYNOMIAL;
		kernelParameter1 = 1;
		kernelParameter3 = 1;
	}

    if (*pSparse > 0)
	{
		psvm->InitializeSparse(
			(void*)pX,
			(size_t*)pVecIndex,	  // sizeof (size_t)==8 <==> as.integer64(bit64 package)
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

	*pClasses = psvm->GetClasses() + 1;

	Rprintf("pMaxIter =%d *pTolerance=%f *pClasses =%d \n", *pMaxIter, *pTolerance, *pClasses );

	unsigned int const repetitions = 256;    // must be a multiple of 16
	for ( unsigned int ii = 0; ii < (unsigned int)(*pMaxIter); ii += repetitions )
	{
		double primal =  std::numeric_limits< double >::infinity();
		double dual   = -std::numeric_limits< double >::infinity();

		try{
			std::pair< CUDA_FLOAT_DOUBLE, CUDA_FLOAT_DOUBLE > const result = psvm->Optimize( repetitions );
			primal = result.first;
			dual   = result.second;

			//Rprintf("Iteration = %d/%d,  primal = %f dual = %f\n", ( ii + 1 ), (*npTotal_iter), primal, dual );
			if ( 2 * ( primal - dual ) < (*pTolerance) * ( primal + dual ) )
				break;
		}
		catch( std::exception& error ) {
			g_error = true;
			g_errorString = error.what();
		}
		catch( ... ) {
			g_error = true;
			g_errorString = "Unknown error";
		}

		*npTotal_iter = ii;

		if(g_error) break;
	}

	if(g_error)
	{
		strcpy( *pszError, g_errorString.c_str() );
		Rprintf("Error: %s", g_errorString.c_str());
		return;
	}

	Rprintf("*npTotal_iter =%d\n", *npTotal_iter );

	boost::shared_array< float > trainingAlphas( new float[ (*pXrow) * (nclasses-1) ] );
	psvm->GetAlphas( (void*)(trainingAlphas.get()), GTSVM_TYPE_FLOAT, columnMajor );

	*pSV = 0;
	int nLableFill = 0;
	for ( unsigned int ii = 0; ii < (unsigned int)(*pXrow); ++ii ) {

		bool zero = true;
		for ( unsigned int jj = 0; jj < (nclasses-1); ++jj ) {

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
				if(pLabels[k]==pY[ii])
				{
					pSVofclass[k] = pSVofclass[k] + 1;
					bFound=true;
				}
			}

			if(!bFound)
			{
				pLabels[nLableFill] = pY[ii];
				pSVofclass[nLableFill] = 1;
				nLableFill++;
			}
		}
	}

	Rprintf("*pSV =%d *pIndex=%d\n", *pSV, *pIndex );

	try{
		psvm->Shrink(smallClusters, activeClusters);
		psvm->GetTrainingVectorNormsSquared( (void*)pTrainingNormsSquared, GTSVM_TYPE_DOUBLE );
		psvm->GetTrainingVectorKernelNormsSquared( (void*)pTrainingNormsSquared, GTSVM_TYPE_DOUBLE );
		psvm->GetTrainingResponses( (void*)pTrainingResponses, GTSVM_TYPE_DOUBLE, columnMajor );
		psvm->GetAlphas( (void*)pTrainingAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
	}
	catch( std::exception& error ) {
		g_error = true;
		g_errorString = error.what();
	}
	catch( ... ) {
		g_error = true;
		g_errorString = "Unknown error";
	}

	if(g_error)
	{
		strcpy( *pszError, g_errorString.c_str() );
		Rprintf("Error: %s", g_errorString.c_str());
		return;
	}
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

	Rprintf("*pKernel_type=%d *nclasses=%d *pModelSparse=%d[%d,%d] First=%f, Last=%f\n", *pKernelType, nclasses, *pModelSparse, *pModelRow, *pModelCol, pModelY[0], pModelY[ *pModelRow - 1] );


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

	try
	{
		psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
		psvm->SetTrainingResponses( (void*)pModelResponses, GTSVM_TYPE_DOUBLE, columnMajor );
		psvm->SetTrainingVectorNormsSquared( (void*)pModelNormsSquared, GTSVM_TYPE_DOUBLE );
		psvm->SetTrainingVectorKernelNormsSquared( (void*)pModelNormsSquared, GTSVM_TYPE_DOUBLE );
		psvm->ClusterTrainingVectors( smallClusters, activeClusters );
	}
	catch( std::exception& error ) {
		g_error = true;
		g_errorString = error.what();
	}
	catch( ... ) {
		g_error = true;
		g_errorString = "Unknown error";
	}
	if(g_error)
	{
		strcpy( *pszError, g_errorString.c_str() );
		Rprintf("Error: %s", g_errorString.c_str());
		return;
	}

	nclasses = psvm->GetClasses();

	boost::shared_array< double > result( new double[ (*pXrow) * nclasses ] );

	Rprintf("*pSparseX=%d *pXrow=%d, nclasses=%d\n", *pSparseX, *pXrow, nclasses);

    if (*pSparseX > 0)
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
	else
		psvm->ClassifyDense(
			(void*)(result.get()),
			GTSVM_TYPE_DOUBLE,
			(void*)pX,
			GTSVM_TYPE_DOUBLE,
			(unsigned)*pXrow,
			(unsigned)*pModelCol,
			columnMajor);

	try
	{
		for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			for ( unsigned int jj = 0; jj < nclasses; ++jj )
				pDec[ ii * nclasses + jj ] = result[ ii * nclasses + jj ];

		if(nclasses==1)
		{
			for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
				if( pDec[ ii ] < 0)  pRet[ ii ]= -1; else pRet[ ii ] = 1;
		}
		else
		{
			for ( unsigned int ii = 0; ii < (unsigned)(*pXrow); ++ii )
			{
				unsigned int n_idx = 0;
				for ( unsigned int jj = 1; jj < nclasses; ++jj )
				{
					if(	pDec[ ii * nclasses + jj ] > pDec[ ii * nclasses + n_idx ] )
						n_idx = jj;
				}

				pRet[ ii ]= n_idx + 1;
			}
		}
	}
	catch( std::exception& error ) {
		g_error = true;
		g_errorString = error.what();
	}
	catch( ... ) {
		g_error = true;
		g_errorString = "Unknown error";
	}

	if(g_error)
	{
		strcpy( *pszError, g_errorString.c_str() );
		Rprintf("Error: %s", g_errorString.c_str());
		return;
	}

	Rprintf("DONE! [%d,%d]\n", *pXrow, nclasses );
}
