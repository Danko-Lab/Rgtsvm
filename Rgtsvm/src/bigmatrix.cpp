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


/*
 * Matrix operations for bigmatrix class
 * since R 3.1, .C(DUP=FALSE) was deprecated, it was out of oder.
 *
 * Will be removed in the future!
 */

extern "C" SEXP set_big_matrix_byrows(SEXP mat, SEXP rows, SEXP value)
{
	SEXP dim = getAttrib( mat, R_DimSymbol );
	int ncol = INTEGER(dim)[0];
	int nrow = INTEGER(dim)[1];

	//SEXP dim_value = getAttrib( value, R_DimSymbol );
	//int ncol_value = INTEGER(dim_value)[0];
	//int nrow_value = INTEGER(dim_value)[1];

	int nlen = XLENGTH(rows);

    double *pMat = REAL(mat);
    double *pVal = REAL(value);
    int *pRows = INTEGER(rows);

    for(int r = 0; r < nlen; r++)
    {
		int idx_row = pRows[r];
      	for(int c = 0; c < ncol; c++)
		 	pMat[idx_row + c * nrow] = pVal[r + c * nrow];
    }

    return R_NilValue;
}


extern "C" void bigmatrix_set_bycols (
		   double *pMat,
		   int* pNRow,
		   int* pNCol,
		   int* pCols,
		   int* pNLen,
		   double *pValue)
{
	// Rprintf("Address=%x nrow=%d, ncol=%d, nlen=%d\n", pMat, *pNRow, *pNCol, *pNLen);

	int nrow = *pNRow;

    for(int c = 0; c < *pNLen; c++)
    {
		int idx_col = pCols[c] - 1;
      	for(int r = 0; r < nrow; r++)
      	{
		 	pMat[r + idx_col * nrow] = pValue[r + c * nrow];
		}
    }

    return;
}


extern "C" void bigmatrix_set_byrows (
		   double *pMat,
		   int* pNRow,
		   int* pNCol,
		   int* pRows,
		   int* pNLen,
		   double *pValue)
{
	// Rprintf("Address=%x nrow=%d, ncol=%d, nlen=%d\n", pMat, *pNRow, *pNCol, *pNLen);

	int ncol = *pNCol;
	int nrow = *pNRow;

    for(int r = 0; r < *pNLen; r++)
    {
		int idx_row = pRows[r] - 1;
		int address = 0;
      	for(int c = 0; c < ncol; c++)
      	{
			address = c * nrow;
		 	pMat[idx_row + address ] = pValue[ r + address];
		}
    }

    return;
}


extern "C" void bigmatrix_exchange_rows (
		   double *pMat,
		   int* pNRow,
		   int* pNCol,
		   int* pNRows1,
		   int* pNLen1,
		   int* pNRows2,
		   int* pNLen2 )
{
	int nrow = *pNRow;
	int ncol = *pNCol;

	for(int k=0; k<*pNLen1; k++)
	{
		int row1 = pNRows1[k] -1;
		int row2 = pNRows2[k] -1;

		for(int c = 0; c < ncol; c++)
		{
			int address = c * nrow;
			double t = pMat[ row2 + address];
			pMat[ row2 + address] = pMat[row1 + address];
			pMat[ row1 + address] = t;
		}
	}
    return;
}

extern "C" void bigmatrix_exchange_cols (
		   double *pMat,
		   int* pNRow,
		   int* pNCol,
		   int* pNCols1,
		   int* pNLen1,
		   int* pNCols2,
		   int* pNLen2 )
{
	int nrow = *pNRow;
	//int ncol = *pNCol;

	for(int k=0; k<*pNLen1; k++)
	{
		int col1 = pNCols1[k] -1;
		int col2 = pNCols2[k] -1;

		for(int r = 0; r < nrow; r++)
		{
			double t = pMat[ r + col1 * nrow];
			pMat[ r + col2 * nrow] = pMat[r + col1 * nrow];
			pMat[ r + col1 * nrow] = t;
		}
	}
    return;
}
