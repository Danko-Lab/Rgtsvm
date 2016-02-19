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
	\file cuda_sort.hpp
	\brief Front-end for CUDA sorting kernel
*/




#ifndef __CUDA_SORT_HPP__
#define __CUDA_SORT_HPP__

#ifdef __cplusplus




#include <boost/cstdint.hpp>

#include <utility>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    FindLargest functions
//============================================================================


/*
	destroys both the deviceWork and deviceSource arrays, returns a pointer to
	one of these arrays, containing the result in the first 2^logBatchSize
	elements
*/


std::pair< std::pair< boost::uint32_t const*, boost::uint32_t const* >, unsigned int > UUFindLargest(
	void* deviceWork1,
	void* deviceWork2,
	boost::uint32_t* deviceSourceKeys,
	boost::uint32_t* deviceSourceValues,
	unsigned int const workSize,           // in bytes
	unsigned int const resultSize,         // number of maximal keys/values
	unsigned int const destinationSize,    // number of keys/values
	unsigned int const sourceSize          // number of keys/values
);


std::pair< std::pair< float const*, boost::uint32_t const* >, unsigned int > FUFindLargest(
	void* deviceWork1,
	void* deviceWork2,
	float* deviceSourceKeys,
	boost::uint32_t* deviceSourceValues,
	unsigned int const workSize,           // in bytes
	unsigned int const resultSize,         // number of maximal keys/values
	unsigned int const destinationSize,    // number of keys/values
	unsigned int const sourceSize          // number of keys/values
);


#ifdef CUDA_USE_DOUBLE

std::pair< std::pair< double const*, boost::uint32_t const* >, unsigned int > DUFindLargest(
	void* deviceWork1,
	void* deviceWork2,
	double* deviceSourceKeys,
	boost::uint32_t* deviceSourceValues,
	unsigned int const workSize,           // in bytes
	unsigned int const resultSize,         // number of maximal keys/values
	unsigned int const destinationSize,    // number of keys/values
	unsigned int const sourceSize          // number of keys/values
);

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM




#endif    /* __cplusplus */

#endif    /* __CUDA_SORT_HPP__ */
