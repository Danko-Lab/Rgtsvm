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
	\file gtsvm.h
	\brief definition of C interface to SVM class
*/




#ifndef __GTSVM_H__
#define __GTSVM_H__




#ifdef __cplusplus
extern "C" {
#endif    /* __cplusplus */




#include <stdlib.h>




/*============================================================================
	GTSVM_Context typedef
============================================================================*/


typedef unsigned int GTSVM_Context;




/*============================================================================
	GTSVM_Type enumeration
============================================================================*/


typedef enum {

	GTSVM_TYPE_UNKNOWN = 0,

	GTSVM_TYPE_BOOL,

	GTSVM_TYPE_FLOAT,
	GTSVM_TYPE_DOUBLE,

	GTSVM_TYPE_INT8,
	GTSVM_TYPE_INT16,
	GTSVM_TYPE_INT32,
	GTSVM_TYPE_INT64,

	GTSVM_TYPE_UINT8,
	GTSVM_TYPE_UINT16,
	GTSVM_TYPE_UINT32,
	GTSVM_TYPE_UINT64

} GTSVM_Type;




/*============================================================================
	GTSVM_Kernel enumeration
============================================================================*/


typedef enum {

	GTSVM_KERNEL_UNKNOWN = 0,

	GTSVM_KERNEL_POLYNOMIAL,    /* K( x, y ) = ( p1 * <x,y> + p2 )^p3     */
	GTSVM_KERNEL_GAUSSIAN,      /* K( x, y ) = exp( -p1 * || x - y ||^2 ) */
	GTSVM_KERNEL_SIGMOID        /* K( x, y ) = tanh( p1 * <x,y> + p2 )    */

} GTSVM_Kernel;


#ifdef __cplusplus
}    /* extern "C" */
#endif    /* __cplusplus */




#endif    /* __GTSVM_H__ */
