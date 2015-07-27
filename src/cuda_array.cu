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
	\file cuda_array.cu
	\brief CUDA ArrayUpdate function
*/




#include "cuda_helpers.hpp"
#include "helpers.hpp"

#include <boost/assert.hpp>
#include <boost/cstdint.hpp>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    ArrayReadKernel kernel
//============================================================================


template< typename t_Type >
__global__ void ArrayReadKernel(
	t_Type* const destination,
	t_Type const* const source,
	boost::uint32_t const* const indices,
	unsigned int const size
)
{
	for ( unsigned int ii = ( blockIdx.x << 8 ) + threadIdx.x; ii < size; ii += ( gridDim.x << 8 ) )
		destination[ ii ] = source[ indices[ ii ] ];
}




//============================================================================
//    ArrayUpdateKernel kernel
//============================================================================


template< typename t_Type >
__global__ void ArrayUpdateKernel(
	t_Type* const destination,
	t_Type const* const source,
	boost::uint32_t const* const indices,
	unsigned int const size
)
{
	for ( unsigned int ii = ( blockIdx.x << 8 ) + threadIdx.x; ii < size; ii += ( gridDim.x << 8 ) )
		destination[ indices[ ii ] ] = source[ ii ];
}




//============================================================================
//    ArraySetKernel kernel
//============================================================================


template< typename t_Type >
__global__ void ArraySetKernel(
	t_Type* const destination,
	t_Type const source,
	boost::uint32_t const* const indices,
	unsigned int const size
)
{
	for ( unsigned int ii = ( blockIdx.x << 8 ) + threadIdx.x; ii < size; ii += ( gridDim.x << 8 ) )
		destination[ indices[ ii ] ] = source;
}




//============================================================================
//    ArrayReadHelper helper function
//============================================================================


template< typename t_Type >
void ArrayReadHelper(
	t_Type* const deviceDestination,
	t_Type const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = ( ( size + 255 ) >> 8 );
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	ArrayReadKernel< t_Type ><<< blocks, 256 >>>( deviceDestination, deviceSource, deviceIndices, size );
}




//============================================================================
//    ArrayRead functions
//============================================================================


void BArrayRead(
	bool* const deviceDestination,
	bool const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayReadHelper< bool >( deviceDestination, deviceSource, deviceIndices, size );
}


void IArrayRead(
	boost::uint32_t* const deviceDestination,
	boost::uint32_t const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayReadHelper< boost::uint32_t >( deviceDestination, deviceSource, deviceIndices, size );
}


void FArrayRead(
	float* const deviceDestination,
	float const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayReadHelper< float >( deviceDestination, deviceSource, deviceIndices, size );
}


#ifdef CUDA_USE_DOUBLE

void DArrayRead(
	double* const deviceDestination,
	double const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayReadHelper< double >( deviceDestination, deviceSource, deviceIndices, size );
}

#endif    // CUDA_USE_DOUBLE




//============================================================================
//    ArrayUpdateHelper helper function
//============================================================================


template< typename t_Type >
void ArrayUpdateHelper(
	t_Type* const deviceDestination,
	t_Type const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = ( ( size + 255 ) >> 8 );
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	ArrayUpdateKernel< t_Type ><<< blocks, 256 >>>( deviceDestination, deviceSource, deviceIndices, size );
}




//============================================================================
//    ArrayUpdate functions
//============================================================================


void BArrayUpdate(
	bool* const deviceDestination,
	bool const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayUpdateHelper< bool >( deviceDestination, deviceSource, deviceIndices, size );
}


void IArrayUpdate(
	boost::uint32_t* const deviceDestination,
	boost::uint32_t const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayUpdateHelper< boost::uint32_t >( deviceDestination, deviceSource, deviceIndices, size );
}


void FArrayUpdate(
	float* const deviceDestination,
	float const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayUpdateHelper< float >( deviceDestination, deviceSource, deviceIndices, size );
}


#ifdef CUDA_USE_DOUBLE

void DArrayUpdate(
	double* const deviceDestination,
	double const* const deviceSource,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArrayUpdateHelper< double >( deviceDestination, deviceSource, deviceIndices, size );
}

#endif    // CUDA_USE_DOUBLE




//============================================================================
//    ArraySetHelper helper function
//============================================================================


template< typename t_Type >
void ArraySetHelper(
	t_Type* const deviceDestination,
	t_Type const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = ( ( size + 255 ) >> 8 );
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	ArraySetKernel< t_Type ><<< blocks, 256 >>>( deviceDestination, source, deviceIndices, size );
}




//============================================================================
//    ArraySet functions
//============================================================================


void BArraySet(
	bool* const deviceDestination,
	bool const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArraySetHelper< bool >( deviceDestination, source, deviceIndices, size );
}


void IArraySet(
	boost::uint32_t* const deviceDestination,
	boost::uint32_t const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArraySetHelper< boost::uint32_t >( deviceDestination, source, deviceIndices, size );
}


void FArraySet(
	float* const deviceDestination,
	float const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArraySetHelper< float >( deviceDestination, source, deviceIndices, size );
}


#ifdef CUDA_USE_DOUBLE

void DArraySet(
	double* const deviceDestination,
	double const source,
	boost::uint32_t const* const deviceIndices,
	unsigned int const size
)
{
	ArraySetHelper< double >( deviceDestination, source, deviceIndices, size );
}

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM
