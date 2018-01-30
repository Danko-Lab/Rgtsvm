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
#include "Rgtsvm.hpp"


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
		  int	*pnError)
{
	*pnError = 0;
	GTSVM::SVM* psvm = new GTSVM::SVM();
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

	if(*pVerbose) Rprintf("[e-SVR predict#] X=%d[%d,%d] kernel=%d degree=%f gamma=%f, c0=%f C=%f\n",
			*pModelSparse, *pModelRow, *pModelCol, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization );

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
	_CHECK_EXCEPTIONS_RETURN_0_

	if(*pVerbose) Rprintf("[e-SVR predict#] Model=%d[%d,%d] rho=%f\n", *pModelSparse, *pModelRow, *pModelCol, *pModelRho );

	_TRY_EXCEPTIONS_

	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );
	psvm->SetBias(  -1*(*pModelRho) );
	psvm->ClusterTrainingVectors( smallClusters, activeClusters );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_RETURN_0_

	return(psvm);
}

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
		  int	*pnError)
{
	*pnError = 0;
	bool g_error = false;
	std::string g_errorString;
	bool columnMajor = DEF_COLUMN_MAJOR;

	GTSVM::SVM* psvm = (GTSVM::SVM*)pModel;

	if(*pVerbose) Rprintf("[e-SVR predict#] X=%d[%d,] \n", *pSparseX, *pXrow  );

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
			(unsigned)psvm->GetColumns(),
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
			(unsigned)psvm->GetColumns(),
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

	return;
}


extern "C" void* gtsvmpredict_classfication_loadsvm  (
		  int	 *pModelSparse,
		  double *pModelX,
		  int64_t	*pModelVecOffset,
		  int64_t	*pModelVecIndex,
		  int	 *pModelRow,
		  int    *pModelCol,
		  int    *pModelRowIndex,
		  int    *pModelColIndex,

		  int	 *pNclasses,
		  int	 *pTotnSV,
		  double *pModelRho,
		  double *pModelAlphas,

		  int	 *pKernelType,
		  int	 *pDegree,
		  double *pGamma,
		  double *pCoef0,
		  double *pCost,

		  int	*pVerbose,
		  int	*pnError)
{
	*pnError = 0;
	GTSVM::SVM* psvm = new GTSVM::SVM();
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

	if(*pVerbose) Rprintf("[C-SVC predict*] X=%d[%d,%d] nclasses=%d, kernel=%d degree=%f gamma=%f, c0=%f C=%f\n",
			*pModelSparse, *pModelRow, *pModelCol, nclasses, *pKernelType, kernelParameter3, kernelParameter1, kernelParameter2, regularization );

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
	_CHECK_EXCEPTIONS_RETURN_0_

	_TRY_EXCEPTIONS_

	//psvm->ClusterTrainingVectors( smallClusters, activeClusters );
	psvm->SetBias(  -1*(*pModelRho) );
	psvm->SetAlphas( (void*)pModelAlphas, GTSVM_TYPE_DOUBLE, columnMajor );

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_RETURN_0_

	if(*pVerbose) Rprintf("[C-SVC predict*] Model=%d[%d,%d] rho=%f\n", *pModelSparse, *pModelRow, *pModelCol, *pModelRho );

	return(psvm);
}


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
		  int	*pnError)
{
	*pnError = 0;
	bool g_error = false;
	std::string g_errorString;
	bool columnMajor = DEF_COLUMN_MAJOR;

	if(*pVerbose) Rprintf("[C-SVC predict#] X=%d[%d,] \n", *pSparseX, *pXrow  );


	GTSVM::SVM* psvm = (GTSVM::SVM*)pModel;
	unsigned int nclasses = psvm->GetClasses();
	bool multiclass = (nclasses > 2 );
    unsigned int ncol = 1;
    if (nclasses>2) ncol = nclasses;

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
			(unsigned)psvm->GetColumns(),
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
			(unsigned)psvm->GetColumns(),
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



extern "C" void gtsvmpredict_unloadsvm_C( void *pModel, int *pnError  )
{
	bool g_error = false;
	std::string g_errorString;
	*pnError = 0;

	_TRY_EXCEPTIONS_

	GTSVM::SVM* psvm = (GTSVM::SVM*)pModel;
	delete(psvm);

	_CATCH_EXCEPTIONS_
	_CHECK_EXCEPTIONS_
}

// deviceID starts from 0...nGPU-1
extern "C" int gtsvm_selectDevice( int deviceID, int* npTotal )
{
	if ( cudaGetDeviceCount ( npTotal ) != cudaSuccess )
		return(-1);

	if( deviceID <0 )
	{
		if ( cudaSetDevice ( 0 ) != cudaSuccess )
			return(-1);
	}
	else
	{
		if ( cudaSetDevice ( deviceID ) != cudaSuccess )
			return(-1);
	}

	return(0);

}

extern "C" void gtsvm_resetDevice()
{
	cudaDeviceReset();
}

