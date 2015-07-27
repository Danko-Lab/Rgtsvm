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
	\file cuda_partial_sum.hpp
	\brief Front-end for CUDA partial sum kernel
*/




#ifndef __CUDA_PARTIAL_SUM_HPP__
#define __CUDA_PARTIAL_SUM_HPP__

#ifdef __cplusplus




#include <boost/cstdint.hpp>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    PartialSum function
//============================================================================


// result has one more element than input ((1,2,3,4)->(0,1,3,6,10))
void PartialSum(
	boost::uint32_t* const deviceData,
	void* const deviceWork,
	unsigned int const workSize,     // bytes
	unsigned int const sourceSize    // elements
);




//============================================================================
//    FindNonzeroIndices functions
//============================================================================


unsigned int BFindNonzeroIndices(
	boost::uint32_t* const deviceDestination,
	void* const deviceWork,
	bool const* const deviceSource,
	unsigned int const workSize,     // bytes
	unsigned int const sourceSize    // elements
);


unsigned int FFindNonzeroIndices(
	boost::uint32_t* const deviceDestination,
	void* const deviceWork,
	float const* const deviceSource,
	unsigned int const workSize,     // bytes
	unsigned int const sourceSize    // elements
);


#ifdef CUDA_USE_DOUBLE

unsigned int DFindNonzeroIndices(
	boost::uint32_t* const deviceDestination,
	void* const deviceWork,
	double const* const deviceSource,
	unsigned int const workSize,     // bytes
	unsigned int const sourceSize    // elements
);

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM




#endif    /* __cplusplus */

#endif    /* __CUDA_PARTIAL_SUM_HPP__ */
