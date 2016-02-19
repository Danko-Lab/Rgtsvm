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
	\file cuda_array.hpp
	\brief Front-end for CUDA ArrayRead and ArrayUpdate functions
*/




#ifndef __CUDA_ARRAY_HPP__
#define __CUDA_ARRAY_HPP__

#ifdef __cplusplus




#include <boost/cstdint.hpp>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    ArrayRead functions
//============================================================================


void BArrayRead(
	bool* const deviceDestination,
	bool const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


void IArrayRead(
	boost::uint32_t* const deviceDestination,
	boost::uint32_t const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


void FArrayRead(
	float* const deviceDestination,
	float const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


#ifdef CUDA_USE_DOUBLE

void DArrayRead(
	double* const deviceDestination,
	double const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);

#endif    // CUDA_USE_DOUBLE




//============================================================================
//    ArrayUpdate functions
//============================================================================


void BArrayUpdate(
	bool* const deviceDestination,
	bool const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


void IArrayUpdate(
	boost::uint32_t* const deviceDestination,
	boost::uint32_t const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


void FArrayUpdate(
	float* const deviceDestination,
	float const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


#ifdef CUDA_USE_DOUBLE

void DArrayUpdate(
	double* const deviceDestination,
	double const* const deviceValues,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);

#endif    // CUDA_USE_DOUBLE




//============================================================================
//    ArraySet functions
//============================================================================


void BArraySet(
	bool* const deviceDestination,
	bool const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


void IArraySet(
	boost::uint32_t* const deviceDestination,
	boost::uint32_t const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


void FArraySet(
	float* const deviceDestination,
	float const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);


#ifdef CUDA_USE_DOUBLE

void DArraySet(
	double* const deviceDestination,
	double const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
);

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM




#endif    /* __cplusplus */

#endif    /* __CUDA_ARRAY_HPP__ */
