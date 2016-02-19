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
	\file cuda_reduce.hpp
	\brief Front-end for CUDA reduction kernel
*/




#ifndef __CUDA_REDUCE_HPP__
#define __CUDA_REDUCE_HPP__

#ifdef __cplusplus




#include <boost/cstdint.hpp>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    Reduce functions
//============================================================================


/*
	destroys both the deviceWork and deviceSource arrays, returns a pointer to
	one of these arrays, containing the result in the first 2^logBatchSize
	elements
*/


boost::uint32_t const* UReduce(
	void* deviceWork,
	boost::uint32_t* deviceSource,
	unsigned int const workSize,       // bytes
	unsigned int const sourceSize,     // elements
	unsigned int const sourcePitch,    // elements
	unsigned int const copies = 0
);


float const* FReduce(
	void* deviceWork,
	float* deviceSource,
	unsigned int const workSize,       // bytes
	unsigned int const sourceSize,     // elements
	unsigned int const sourcePitch,    // elements
	unsigned int const copies = 0
);


#ifdef CUDA_USE_DOUBLE

double const* DReduce(
	void* deviceWorkVoid,
	double* deviceSource,
	unsigned int const workSize,       // bytes
	unsigned int const sourceSize,     // elements
	unsigned int const sourcePitch,    // elements
	unsigned int const copies = 0
);

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM




#endif    /* __cplusplus */

#endif    /* __CUDA_REDUCE_HPP__ */
