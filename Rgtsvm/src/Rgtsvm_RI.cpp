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
	\file Rgtsvm_RI.cpp
	\brief R interface of svm and predict
*/

#include "headers.hpp"
#include <R.h>
#include <Rinternals.h>
#include <Rembedded.h>
#include <Rdefines.h>
#include <R_ext/Parse.h>
#include "Rgtsvm.hpp"


extern "C" SEXP gtsvmtrain_epsregression (
		SEXP spSparse,
		SEXP spX,
		SEXP spVecOffset,// start from 0
		SEXP spVecIndex, // start from 1
		SEXP spXrow,
		SEXP spXcol,
		SEXP spXInnerRow,
		SEXP spXInnerCol,
		SEXP spXRowIndex,
		SEXP spXColIndex,
		SEXP spY,

		SEXP spKernelType,
		SEXP spDegree,
		SEXP spGamma,
		SEXP spCoef0,
		SEXP spCost,
		SEXP spTolerance,
		SEXP spEpsilon,
		SEXP spShrinking,
		SEXP spFitted,
		SEXP spMaxIter,
		SEXP spNoProgressIgnore,
		SEXP spVerbose)
{
	int nRow = INTEGER(spXrow)[0];

	SEXP spRet,t;
   	PROTECT(spRet = t = allocList(9+1));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	//# total iteration
	SEXP spTotalIter;
    PROTECT( spTotalIter = allocVector( INTSXP, 1) );
	SETCAR( t, spTotalIter );
	SET_TAG(t, install("totalIter") );
	t = CDR(t);

	//# the total number of support vectors
	SEXP spSV;
    PROTECT( spSV = allocVector( INTSXP, 1) );
	SETCAR( t, spSV );
	SET_TAG(t, install("nr") );
	t = CDR(t);

	//# the labels of classes
	SEXP spLabel;
    PROTECT( spLabel = allocVector( INTSXP, 1) );
	SETCAR( t, spLabel );
	SET_TAG(t, install("labels") );
	t = CDR(t);

	//# the support vectors of each classes
	SEXP spSVofclass;
    PROTECT( spSVofclass = allocVector( INTSXP, 1) );
	SETCAR( t, spSVofclass );
	SET_TAG(t, install("nSV") );
	t = CDR(t);

	//# - m_bias
	SEXP spRho;
    PROTECT( spRho = allocVector( REALSXP, 1) );
	SETCAR( t, spRho );
	SET_TAG(t, install("rho") );
	t = CDR(t);

	//# the index of support vectors
	SEXP spIndex;
    PROTECT( spIndex = allocVector( INTSXP, nRow) );
	SETCAR( t, spIndex );
	SET_TAG(t, install("index") );
	t = CDR(t);

	//# dont alpha value for each classes and support vector dim[nr, nclass-1]
	SEXP spTrainingAlphas;
    PROTECT( spTrainingAlphas = allocVector( REALSXP, nRow) );
	SETCAR( t, spTrainingAlphas );
	SET_TAG(t, install("coefs") );
	t = CDR(t);

	//# prdict labels for the fitted option
	SEXP spPredict;
    PROTECT( spPredict = allocVector( REALSXP, nRow) );
	SETCAR( t, spPredict );
	SET_TAG(t, install("predict") );
	t = CDR(t);

	gtsvmtrain_epsregression_C (
		INTEGER( spSparse ),
		REAL( spX ),
		(int64_t*)REAL( spVecOffset ),
		(int64_t*)REAL( spVecIndex ),
		INTEGER( spXrow ),
		INTEGER( spXcol ),
		INTEGER( spXInnerRow ),
		INTEGER( spXInnerCol ),
		INTEGER( spXRowIndex ),
		INTEGER( spXColIndex ),
		REAL( spY ),

		INTEGER( spKernelType ),
		INTEGER( spDegree ),
		REAL( spGamma ),
		REAL( spCoef0 ),
		REAL( spCost ),
		REAL( spTolerance ),
		REAL( spEpsilon ),
		INTEGER( spShrinking ),
		INTEGER( spFitted ),
		INTEGER( spMaxIter ),
		INTEGER( spNoProgressIgnore ),

	    INTEGER( spTotalIter ),
		INTEGER( spSV ),
		INTEGER( spIndex ),
		INTEGER( spLabel ),
		INTEGER( spSVofclass ),
		REAL( spRho ),
        REAL( spTrainingAlphas ),
		REAL( spPredict ),
		INTEGER( spVerbose ),
		INTEGER( spError ) );

	UNPROTECT(10);

    return( spRet );
}

extern "C" SEXP gtsvmpredict_epsregression  (
		SEXP spDecisionvalues,
		SEXP spProbability,

		SEXP spModelSparse,
		SEXP spModelX,
		SEXP spModelVecOffset,
		SEXP spModelVecIndex,
		SEXP spModelRow,
		SEXP spModelCol,
		SEXP spModelRowIndex,
		SEXP spModelColIndex,

		SEXP spTotnSV,
		SEXP spModelRho,
		SEXP spModelAlphas,

		SEXP spKernelType,
		SEXP spDegree,
		SEXP spGamma,
		SEXP spCoef0,
		SEXP spCost,

		SEXP spSparseX,
		SEXP spX,
		SEXP spXVecOffset,
		SEXP spXVecIndex,
		SEXP spXrow,
		SEXP spXInnerRow,
		SEXP spXInnerCol,
		SEXP spXRowIndex,
		SEXP spXColIndex,
		SEXP spVerbose)
{
	int nRow = INTEGER(spXrow)[0];

	SEXP spRet, t;
   	PROTECT(spRet = t = allocList(4+1));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	SEXP spRRet;
    PROTECT( spRRet = allocVector( REALSXP, nRow) );
	SETCAR( t, spRRet );
	SET_TAG(t, install("ret") );
	t = CDR(t);

	SEXP spDec;
    PROTECT( spDec = allocVector( REALSXP, nRow) );
	SETCAR( t, spDec );
	SET_TAG(t, install("dec") );
	t = CDR(t);

	SEXP spProb;
    PROTECT( spProb = allocVector( REALSXP, nRow) );
	SETCAR( t, spProb );
	SET_TAG(t, install("prob") );
	t = CDR(t);

	gtsvmpredict_epsregression_C  (
		INTEGER( spDecisionvalues ),
		INTEGER( spProbability ),

		INTEGER( spModelSparse ),
		REAL( spModelX ),
		(int64_t*)REAL( spModelVecOffset ),
		(int64_t*)REAL( spModelVecIndex ),
		INTEGER( spModelRow ),
		INTEGER( spModelCol ),
		INTEGER( spModelRowIndex ),
		INTEGER( spModelColIndex ),

		INTEGER( spTotnSV ),
		REAL( spModelRho ),
		REAL( spModelAlphas ),

		INTEGER( spKernelType ),
		INTEGER( spDegree ),
		REAL( spGamma ),
		REAL( spCoef0 ),
		REAL( spCost ),

		INTEGER( spSparseX ),
		REAL( spX ),
		(int64_t*)REAL( spXVecOffset ),
		(int64_t*)REAL( spXVecIndex ),
		INTEGER( spXrow ),
		INTEGER( spXInnerRow ),
		INTEGER( spXInnerCol ),
		INTEGER( spXRowIndex ),
		INTEGER( spXColIndex ),

		REAL( spRRet ),
		REAL( spDec ),
		REAL( spProb ),
		INTEGER( spVerbose ),
		INTEGER( spError) );

	UNPROTECT(5);

	return(spRet);
}

extern "C" SEXP gtsvmtrain_classfication (
		SEXP spSparse,
		SEXP spX,
		SEXP spVecOffset,// start from 0
		SEXP spVecIndex, // start from 1
		SEXP spXrow,
		SEXP spXcol,
		SEXP spXInnerRow,
		SEXP spXInnerCol,
		SEXP spXRowIndex,
		SEXP spXColIndex,
		SEXP spY,

		SEXP spKernelType,
		// the number of classes
		SEXP spRclasses,
		// KernelParameter3 <==> degree in libsvm
		SEXP spDegree,
		// KernelParameter1 <==> gama in libsvm
		SEXP spGamma,
		// KernelParameter2 <==> coef0 in libsvm
		SEXP spCoef0,
		// pRegularization <==> cost in libsvm
		SEXP spCost,
		SEXP spClassWeight,
		SEXP spTolerance,
		SEXP spFitted,
		SEXP spBiased,
		SEXP spMaxIter,
		SEXP spNoProgressIgnore,
		SEXP spNclass,
		SEXP spVerbose)
{
	int nRow = INTEGER(spXrow)[0];
	int nclass = INTEGER(spNclass)[0];

	SEXP spRet, t;
   	PROTECT(spRet = t = allocList(10+1));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	SEXP spClasses;
    PROTECT( spClasses = allocVector( INTSXP, 1) );
	SETCAR( t, spClasses );
	SET_TAG(t, install("nclasses") );
	t = CDR(t);

	//# total iteration
	SEXP spTotalIter;
    PROTECT( spTotalIter = allocVector( INTSXP, 1) );
	SETCAR( t, spTotalIter );
	SET_TAG(t, install("totalIter") );
	t = CDR(t);

	//# the total number of support vectors
	SEXP spSV;
    PROTECT( spSV = allocVector( INTSXP, 1) );
	SETCAR( t, spSV );
	SET_TAG(t, install("nr") );
	t = CDR(t);

	//# the labels of classes
	SEXP spLabel;
    PROTECT( spLabel = allocVector( INTSXP, nclass) );
	SETCAR( t, spLabel );
	SET_TAG(t, install("labels") );
	t = CDR(t);

	//# the support vectors of each classes
	SEXP spSVofclass;
    PROTECT( spSVofclass = allocVector( INTSXP, nclass) );
	SETCAR( t, spSVofclass );
	SET_TAG(t, install("nSV") );
	t = CDR(t);

	//# - m_bias
	SEXP spRho;
    PROTECT( spRho = allocVector( REALSXP, nclass*(nclass-1)/2) );
	SETCAR( t, spRho );
	SET_TAG(t, install("rho") );
	t = CDR(t);

	//# the index of support vectors
	SEXP spIndex;
    PROTECT( spIndex = allocVector( INTSXP, nRow) );
	SETCAR( t, spIndex );
	SET_TAG(t, install("index") );
	t = CDR(t);

	//# dont alpha value for each classes and support vector dim[nr, nclass-1]
	SEXP spTrainingAlphas;
    PROTECT( spTrainingAlphas = allocVector( REALSXP, nRow*nclass) );
	SETCAR( t, spTrainingAlphas );
	SET_TAG(t, install("coefs") );
	t = CDR(t);

	//# prdict labels for the fitted option
	SEXP spPredict;
    PROTECT( spPredict = allocVector( REALSXP, nRow) );
	SETCAR( t, spPredict );
	SET_TAG(t, install("predict") );
	t = CDR(t);

	//# prdict labels for the fitted option
	SEXP spDecision;
    PROTECT( spDecision = allocVector( REALSXP, nRow*nclass) );
	SETCAR( t, spDecision );
	SET_TAG(t, install("decision") );
	t = CDR(t);

	gtsvmtrain_classfication_C (
		INTEGER( spSparse ),
		REAL( spX ),
		(int64_t*)REAL( spVecOffset ),// start from 0
		(int64_t*)REAL( spVecIndex ), // start from 1
		INTEGER( spXrow ),
		INTEGER( spXcol ),
		INTEGER( spXInnerRow ),
		INTEGER( spXInnerCol ),
		INTEGER( spXRowIndex ),
		INTEGER( spXColIndex ),
		REAL( spY),

		INTEGER( spKernelType ),
		INTEGER( spRclasses ),
		INTEGER( spDegree ),
		REAL( spGamma ),
		REAL( spCoef0 ),
		REAL( spCost ),
		REAL( spClassWeight ),
		REAL( spTolerance ),
		INTEGER( spFitted ),
		INTEGER( spBiased ),
		INTEGER( spMaxIter ),
		INTEGER( spNoProgressIgnore ),

		//output variables
		//# total iteration
		INTEGER( spTotalIter ),
		//# the total number of classes
		INTEGER( spClasses ),
		//# the total number of support vectors
		INTEGER( spSV ),
		//# the index of support vectors
		INTEGER( spIndex ),
		//# the labels of classes
		INTEGER( spLabel ),
		//# the support vectors of each classes
		INTEGER( spSVofclass ),
		//# dont know
		REAL( spRho ),
		//# dont alpha value for each classes and support vector dim[nr, nclass-1]
		REAL( spTrainingAlphas ),
		REAL( spPredict ),
		REAL( spDecision ),
		INTEGER( spVerbose ),
		INTEGER( spError ) );

	UNPROTECT(12);

	return(spRet);
}

extern "C" SEXP gtsvmpredict_classfication  (
		SEXP spDecisionvalues,
		SEXP spProbability,
		SEXP spModelSparse,
		SEXP spModelX,
		SEXP spModelVecOffset,
		SEXP spModelVecIndex,
		SEXP spModelRow,
		SEXP spModelCol,
		SEXP spModelRowIndex,
		SEXP spModelColIndex,

		SEXP spNclasses,
		SEXP spTotnSV,
		SEXP spModelRho,
		SEXP spModelAlphas,

		SEXP spKernelType,
		SEXP spDegree,
		SEXP spGamma,
		SEXP spCoef0,
		SEXP spCost,

		SEXP spSparseX,
		SEXP spX,
		SEXP spXVecOffset,
		SEXP spXVecIndex,
		SEXP spXrow,
		SEXP spXInnerRow,
		SEXP spXInnerCol,
		SEXP spXRowIndex,
		SEXP spXColIndex,
		SEXP spVerbose)
{
	int nRow = INTEGER(spXrow)[0];
	int nclass = INTEGER(spNclasses)[0];

	SEXP spRet,t;
   	PROTECT(spRet = t = allocList(4+1));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	SEXP spRRet;
    PROTECT( spRRet = allocVector( REALSXP, nRow) );
	SETCAR( t, spRRet );
	SET_TAG(t, install("ret") );
	t = CDR(t);

	SEXP spDec;
    PROTECT( spDec = allocVector( REALSXP, nRow*nclass) );
	SETCAR( t, spDec );
	SET_TAG(t, install("dec") );
	t = CDR(t);

	SEXP spProb;
    PROTECT( spProb = allocVector( REALSXP, nRow*nclass) );
	SETCAR( t, spProb );
	SET_TAG(t, install("prob") );
	t = CDR(t);

	gtsvmpredict_classfication_C  (
		INTEGER( spDecisionvalues ),
		INTEGER( spProbability ),
		INTEGER( spModelSparse ),
		REAL( spModelX ),
		(int64_t*)REAL( spModelVecOffset ),
		(int64_t*)REAL( spModelVecIndex ),
		INTEGER( spModelRow ),
		INTEGER( spModelCol ),
		INTEGER( spModelRowIndex ),
		INTEGER( spModelColIndex ),

		INTEGER( spNclasses ),
		INTEGER( spTotnSV ),
		REAL( spModelRho ),
		REAL( spModelAlphas ),

		INTEGER( spKernelType ),
		INTEGER( spDegree ),
		REAL( spGamma ),
		REAL( spCoef0 ),
		REAL( spCost ),

		INTEGER( spSparseX ),
		REAL( spX ),
		(int64_t*)REAL( spXVecOffset ),
		(int64_t*)REAL( spXVecIndex ),
		INTEGER( spXrow ),
		INTEGER( spXInnerRow ),
		INTEGER( spXInnerCol ),
		INTEGER( spXRowIndex ),
		INTEGER( spXColIndex ),

		REAL( spRRet ),
		REAL( spDec ),
		REAL( spProb ),
		INTEGER( spVerbose ),
		INTEGER( spError) );

	UNPROTECT(5);

	return(spRet);

}

extern "C" SEXP gtsvmpredict_loadsvm  (
		SEXP spModelSparse,
		SEXP spModelX,
		SEXP spModelVecOffset,
		SEXP spModelVecIndex,
		SEXP spModelRow,
		SEXP spModelCol,
		SEXP spModelRowIndex,
		SEXP spModelColIndex,

		SEXP spNclasses,
		SEXP spTotnSV,
		SEXP spModelRho,
		SEXP spModelAlphas,

		SEXP spKernelType,
		SEXP spDegree,
		SEXP spGamma,
		SEXP spCoef0,
		SEXP spCost,

		SEXP spVerbose)
{
	SEXP spRet,t;
   	PROTECT(spRet = t = allocList(3));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	int nclass = INTEGER( spNclasses )[0];
	void* p = NULL;

	if( nclass!=0)
	{
		p = gtsvmpredict_classfication_loadsvm (
			INTEGER( spModelSparse ),
			REAL( spModelX ),
			(int64_t*)REAL( spModelVecOffset ),
			(int64_t*)REAL( spModelVecIndex ),
			INTEGER( spModelRow ),
			INTEGER( spModelCol ),
			INTEGER( spModelRowIndex ),
			INTEGER( spModelColIndex ),

			INTEGER( spNclasses ),
			INTEGER( spTotnSV ),
			REAL( spModelRho ),
			REAL( spModelAlphas ),

			INTEGER( spKernelType ),
			INTEGER( spDegree ),
			REAL( spGamma ),
			REAL( spCoef0 ),
			REAL( spCost ),

			INTEGER( spVerbose ),
			INTEGER( spError) );
	}
	else
	{
		p = gtsvmpredict_epsregression_loadsvm  (
			INTEGER( spModelSparse ),
			REAL( spModelX ),
			(int64_t*)REAL( spModelVecOffset ),
			(int64_t*)REAL( spModelVecIndex ),
			INTEGER( spModelRow ),
			INTEGER( spModelCol ),
			INTEGER( spModelRowIndex ),
			INTEGER( spModelColIndex ),

			INTEGER( spTotnSV ),
			REAL( spModelRho ),
			REAL( spModelAlphas ),

			INTEGER( spKernelType ),
			INTEGER( spDegree ),
			REAL( spGamma ),
			REAL( spCoef0 ),
			REAL( spCost ),

			INTEGER( spVerbose ),
			INTEGER( spError) );
	}

	SEXP spPointer;
	PROTECT( spPointer = R_MakeExternalPtr(p, install("pointer"), R_NilValue) );
	SETCAR( t, spPointer );
	SET_TAG(t, install("pointer") );

	UNPROTECT(3);

	return(spRet);
}

extern "C" SEXP gtsvmpredict_unloadsvm(	SEXP spModelPointer, SEXP pVerbose)
{
	SEXP spRet,t;
   	PROTECT(spRet = t = allocList(3));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	if (TYPEOF(spModelPointer) == EXTPTRSXP)
		gtsvmpredict_unloadsvm_C( EXTPTR_PTR(spModelPointer), INTEGER( spError ) );
	else
		INTEGER(spError)[0] = -1;

	UNPROTECT(2);

	return( spRet);
}

extern "C" SEXP gtsvmpredict_epsregression_direct  (
		SEXP spDecisionvalues,
		SEXP spProbability,

	 	SEXP spModelPointer,

		SEXP spSparseX,
		SEXP spX,
		SEXP spXVecOffset,
		SEXP spXVecIndex,
		SEXP spXrow,
		SEXP spXInnerRow,
		SEXP spXInnerCol,
		SEXP spXRowIndex,
		SEXP spXColIndex,
		SEXP spVerbose)
{
	int nRow = INTEGER(spXrow)[0];

	SEXP spRet, t;
   	PROTECT(spRet = t = allocList(4+1));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	SEXP spRRet;
    PROTECT( spRRet = allocVector( REALSXP, nRow) );
	SETCAR( t, spRRet );
	SET_TAG(t, install("ret") );
	t = CDR(t);

	SEXP spDec;
    PROTECT( spDec = allocVector( REALSXP, nRow) );
	SETCAR( t, spDec );
	SET_TAG(t, install("dec") );
	t = CDR(t);

	SEXP spProb;
    PROTECT( spProb = allocVector( REALSXP, nRow) );
	SETCAR( t, spProb );
	SET_TAG(t, install("prob") );
	t = CDR(t);

	if (TYPEOF(spModelPointer) != EXTPTRSXP)
		INTEGER(spError)[0] = -1;
	else
		gtsvmpredict_epsregression_direct_C  (
			(void*)EXTPTR_PTR(spModelPointer),

			INTEGER( spDecisionvalues ),
			INTEGER( spProbability ),

			INTEGER( spSparseX ),
			REAL( spX ),
			(int64_t*)REAL( spXVecOffset ),
			(int64_t*)REAL( spXVecIndex ),
			INTEGER( spXrow ),
			INTEGER( spXInnerRow ),
			INTEGER( spXInnerCol ),
			INTEGER( spXRowIndex ),
			INTEGER( spXColIndex ),

			REAL( spRRet ),
			REAL( spDec ),
			REAL( spProb ),
			INTEGER( spVerbose ),
			INTEGER( spError) );

	UNPROTECT(5);

	return(spRet);
}

extern "C" SEXP gtsvmpredict_classfication_direct  (
		SEXP spDecisionvalues,
		SEXP spProbability,

	 	SEXP spModelPointer,

		SEXP spSparseX,
		SEXP spX,
		SEXP spXVecOffset,
		SEXP spXVecIndex,
		SEXP spXrow,
		SEXP spXInnerRow,
		SEXP spXInnerCol,
		SEXP spXRowIndex,
		SEXP spXColIndex,
		SEXP spVerbose)
{
	GTSVM::SVM* psvm = (GTSVM::SVM*) ( EXTPTR_PTR(spModelPointer) );
	int nclass = psvm->GetClasses();
	int nRow = INTEGER(spXrow)[0];

	SEXP spRet,t;
   	PROTECT(spRet = t = allocList(4+1));

	SEXP spError;
    PROTECT( spError = allocVector( INTSXP, 1) );
	SETCAR( t, spError );
	SET_TAG(t, install("error") );
	t = CDR(t);

	SEXP spRRet;
    PROTECT( spRRet = allocVector( REALSXP, nRow) );
	SETCAR( t, spRRet );
	SET_TAG(t, install("ret") );
	t = CDR(t);

	SEXP spDec;
    PROTECT( spDec = allocVector( REALSXP, nRow*nclass) );
	SETCAR( t, spDec );
	SET_TAG(t, install("dec") );
	t = CDR(t);

	SEXP spProb;
    PROTECT( spProb = allocVector( REALSXP, nRow*nclass) );
	SETCAR( t, spProb );
	SET_TAG(t, install("prob") );
	t = CDR(t);

	if (TYPEOF(spModelPointer) != EXTPTRSXP)
		INTEGER(spError)[0] = -1;
	else
		gtsvmpredict_classfication_direct_C  (
			(void*)EXTPTR_PTR(spModelPointer),

			INTEGER( spDecisionvalues ),
			INTEGER( spProbability ),

			INTEGER( spSparseX ),
			REAL( spX ),
			(int64_t*)REAL( spXVecOffset ),
			(int64_t*)REAL( spXVecIndex ),
			INTEGER( spXrow ),
			INTEGER( spXInnerRow ),
			INTEGER( spXInnerCol ),
			INTEGER( spXRowIndex ),
			INTEGER( spXColIndex ),

			REAL( spRRet ),
			REAL( spDec ),
			REAL( spProb ),
			INTEGER( spVerbose ),
			INTEGER( spError) );

	UNPROTECT(5);

	return(spRet);

}