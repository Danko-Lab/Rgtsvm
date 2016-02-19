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
	\file cuda_partial_sum.cu
	\brief CUDA partial sum kernel
*/




#include "cuda_helpers.hpp"
#include "helpers.hpp"

#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

#include <algorithm>
#include <stdexcept>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    ADJUST_INDEX helper macro
//============================================================================


#define ADJUST_INDEX( index ) ( ( index ) + ( ( index ) >> 4 ) )




//============================================================================
//    PartialSumInnerKernel kernel
//============================================================================


// adapted from the CUDA parallel scan example
__global__ void PartialSumInnerKernel(
	boost::uint32_t* const outerSums,    // size is ceil( size / 512 )
	boost::uint32_t* const innerSums,
	unsigned int const size
)
{
	__shared__ boost::uint32_t sums[ ADJUST_INDEX( 512 ) ];

	unsigned int const iiIndex = ( blockIdx.x << 9 );
	unsigned int const iiStride = ( gridDim.x << 9 );

	for ( unsigned int ii = iiIndex; ii <= size; ii += iiStride ) {

		unsigned int reversedIndex = 255 - threadIdx.x;
		reversedIndex = ( ( reversedIndex & 0x5555 ) << 1 ) | ( ( reversedIndex & 0xaaaa ) >> 1 );
		reversedIndex = ( ( reversedIndex & 0x3333 ) << 2 ) | ( ( reversedIndex & 0xcccc ) >> 2 );
		reversedIndex = ( ( reversedIndex & 0x0f0f ) << 4 ) | ( ( reversedIndex & 0xf0f0 ) >> 4 );
		reversedIndex = ( ( reversedIndex & 0x00ff ) << 8 ) | ( ( reversedIndex & 0xff00 ) >> 8 );
		reversedIndex >>= 16 - 9;

		unsigned int index = ii + threadIdx.x;
		unsigned int accumulator = 0;
		if ( index < size )
			accumulator = innerSums[ index ];
		sums[ ADJUST_INDEX( reversedIndex + 1 ) ] = accumulator;

		index += 256;
		accumulator = 0;
		if ( index < size )
			accumulator = innerSums[ index ];
		sums[ ADJUST_INDEX( reversedIndex ) ] = accumulator;

		__syncthreads();

		// unrolled reduction
		if ( threadIdx.x < 256 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 256 ) ];
		__syncthreads();
		if ( threadIdx.x < 128 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 128 ) ];
		__syncthreads();
		if ( threadIdx.x < 64 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 64 ) ];
		__syncthreads();
		if ( threadIdx.x < 32 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 32 ) ];
		__syncthreads();
		if ( threadIdx.x < 16 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 16 ) ];
		__syncthreads();
		if ( threadIdx.x < 8 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 8 ) ];
		__syncthreads();
		if ( threadIdx.x < 4 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 4 ) ];
		__syncthreads();
		if ( threadIdx.x < 2 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 2 ) ];
		__syncthreads();

		if ( threadIdx.x == 0 ) {

			outerSums[ ii >> 9 ] = sums[ ADJUST_INDEX( 0 ) ] + sums[ ADJUST_INDEX( 1 ) ];

			sums[ ADJUST_INDEX( 0 ) ] = sums[ ADJUST_INDEX( 1 ) ];
			sums[ ADJUST_INDEX( 1 ) ] = 0;
		}

		// unrolled reduction
		__syncthreads();
		if ( threadIdx.x < 2 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 2 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 2 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 4 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 4 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 4 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 8 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 8 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 8 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 16 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 16 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 16 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 32 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 32 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 32 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 64 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 64 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 64 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 128 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 128 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 128 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 256 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 256 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 256 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}

		__syncthreads();

		index = ii + threadIdx.x;
		if ( index <= size )
			innerSums[ index ] = sums[ ADJUST_INDEX( reversedIndex + 1 ) ];

		index += 256;
		if ( index <= size )
			innerSums[ index ] = sums[ ADJUST_INDEX( reversedIndex ) ];

		__syncthreads();
	}
}




//============================================================================
//    PartialSumOuterKernel kernel
//============================================================================


__global__ void PartialSumOuterKernel(
	boost::uint32_t* const outerSums,    // size is ceil( size / ( 2 ^ 8 ) )
	boost::uint32_t* const innerSums,
	unsigned int const size
)
{
	__shared__ boost::uint32_t sum;

	unsigned int const iiIndex = ( blockIdx.x << 9 );
	unsigned int const iiStride = ( gridDim.x << 9 );

	for ( unsigned int ii = iiIndex; ii <= size; ii += iiStride ) {

		if ( threadIdx.x == 0 )
			sum = outerSums[ ii >> 9 ];

		__syncthreads();

		unsigned int index = ii + threadIdx.x;
		if ( index <= size )
			innerSums[ index ] += sum;

		index += 256;
		if ( index <= size )
			innerSums[ index ] += sum;

		__syncthreads();
	}
}




//============================================================================
//    FindNonzeroIndicesInnerKernel kernel
//============================================================================


// adapted from the CUDA parallel scan example
template< typename t_Type >
__global__ void FindNonzeroIndicesInnerKernel(
	boost::uint32_t* const outerSums,    // size is ceil( size / 512 ) )
	boost::uint32_t* const innerSums,
	t_Type const* const source,
	unsigned int const size
)
{
	__shared__ boost::uint32_t sums[ ADJUST_INDEX( 512 ) ];

	unsigned int const iiIndex = ( blockIdx.x << 9 );
	unsigned int const iiStride = ( gridDim.x << 9 );

	for ( unsigned int ii = iiIndex; ii <= size; ii += iiStride ) {

		unsigned int reversedIndex = 255 - threadIdx.x;
		reversedIndex = ( ( reversedIndex & 0x5555 ) << 1 ) | ( ( reversedIndex & 0xaaaa ) >> 1 );
		reversedIndex = ( ( reversedIndex & 0x3333 ) << 2 ) | ( ( reversedIndex & 0xcccc ) >> 2 );
		reversedIndex = ( ( reversedIndex & 0x0f0f ) << 4 ) | ( ( reversedIndex & 0xf0f0 ) >> 4 );
		reversedIndex = ( ( reversedIndex & 0x00ff ) << 8 ) | ( ( reversedIndex & 0xff00 ) >> 8 );
		reversedIndex >>= 16 - 9;

		unsigned int index = ii + threadIdx.x;
		unsigned int accumulator = 0;
		if ( ( index < size ) && ( source[ index ] != 0 ) )
			accumulator = 1;
		sums[ ADJUST_INDEX( reversedIndex + 1 ) ] = accumulator;

		index += 256;
		accumulator = 0;
		if ( ( index < size ) && ( source[ index ] != 0 ) )
			accumulator = 1;
		sums[ ADJUST_INDEX( reversedIndex ) ] = accumulator;

		__syncthreads();

		// unrolled reduction
		if ( threadIdx.x < 256 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 256 ) ];
		__syncthreads();
		if ( threadIdx.x < 128 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 128 ) ];
		__syncthreads();
		if ( threadIdx.x < 64 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 64 ) ];
		__syncthreads();
		if ( threadIdx.x < 32 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 32 ) ];
		__syncthreads();
		if ( threadIdx.x < 16 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 16 ) ];
		__syncthreads();
		if ( threadIdx.x < 8 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 8 ) ];
		__syncthreads();
		if ( threadIdx.x < 4 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 4 ) ];
		__syncthreads();
		if ( threadIdx.x < 2 )
			sums[ ADJUST_INDEX( threadIdx.x ) ] += sums[ ADJUST_INDEX( threadIdx.x + 2 ) ];
		__syncthreads();

		if ( threadIdx.x == 0 ) {

			outerSums[ ii >> 9 ] = sums[ ADJUST_INDEX( 0 ) ] + sums[ ADJUST_INDEX( 1 ) ];

			sums[ ADJUST_INDEX( 0 ) ] = sums[ ADJUST_INDEX( 1 ) ];
			sums[ ADJUST_INDEX( 1 ) ] = 0;
		}

		// unrolled reduction
		__syncthreads();
		if ( threadIdx.x < 2 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 2 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 2 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 4 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 4 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 4 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 8 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 8 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 8 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 16 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 16 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 16 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 32 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 32 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 32 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 64 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 64 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 64 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 128 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 128 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 128 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}
		__syncthreads();
		if ( threadIdx.x < 256 ) {

			unsigned int const temp = sums[ ADJUST_INDEX( threadIdx.x + 256 ) ];
			sums[ ADJUST_INDEX( threadIdx.x + 256 ) ] = sums[ ADJUST_INDEX( threadIdx.x ) ];
			sums[ ADJUST_INDEX( threadIdx.x ) ] += temp;
		}

		__syncthreads();

		index = ii + threadIdx.x;
		if ( index <= size )
			innerSums[ index ] = sums[ ADJUST_INDEX( reversedIndex + 1 ) ];

		index += 256;
		if ( index <= size )
			innerSums[ index ] = sums[ ADJUST_INDEX( reversedIndex ) ];

		__syncthreads();
	}
}




//============================================================================
//    FindNonzeroIndicesOuterKernel kernel
//============================================================================


template< typename t_Type >
__global__ void FindNonzeroIndicesOuterKernel(
	boost::uint32_t* const destination,
	boost::uint32_t* const innerSums,
	t_Type const* const source,
	unsigned int const size
)
{
	unsigned int const iiIndex = ( blockIdx.x << 8 );
	unsigned int const iiStride = ( gridDim.x << 8 );

	for ( unsigned int ii = iiIndex; ii <= size; ii += iiStride ) {

		unsigned int index = ii + threadIdx.x;
		if ( ( index < size ) && ( source[ index ] != 0 ) )
			destination[ innerSums[ index ] ] = index;
	}
}




//============================================================================
//    PartialSum function
//============================================================================


void PartialSum(
	boost::uint32_t* const deviceData,
	void* const deviceWork,
	unsigned int const workSize,
	unsigned int const sourceSize
)
{
	unsigned int const chunks = ( ( ( sourceSize + 1 ) + 511 ) >> 9 );
	if ( workSize < ( chunks + 1 ) * sizeof( boost::uint32_t ) )
		throw std::runtime_error( "PartialSum: work buffer is too small!" );

	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = chunks;
	{	unsigned int const maximumBlocks = 65535;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	// call the partial sum kernel
	PartialSumInnerKernel<<< blocks, 256 >>>( static_cast< boost::uint32_t* >( deviceWork ), deviceData, sourceSize );

	if ( chunks > 1 ) {

		// recursively find partial sums of static_cast< boost::uint32_t* >( deviceWork )
		PartialSum( static_cast< boost::uint32_t* >( deviceWork ), static_cast< void* >( static_cast< boost::uint32_t* >( deviceWork ) + ( chunks + 1 ) ), workSize - ( chunks + 1 ) * sizeof( boost::uint32_t ), chunks );

		// call the addition kernel
		PartialSumOuterKernel<<< blocks, 256 >>>( static_cast< boost::uint32_t* >( deviceWork ), deviceData, sourceSize );
	}
}




//============================================================================
//    FindNonzeroIndicesHelper helper function
//============================================================================


template< typename t_Type >
unsigned int FindNonzeroIndicesHelper(
	boost::uint32_t* const deviceDestination,
	void* const deviceWork,
	t_Type const* const deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize
)
{
	if ( workSize < ( sourceSize + 1 ) * sizeof( boost::uint32_t ) )
		throw std::runtime_error( "FindNonzeroIndices: work buffer is too small!" );

	unsigned int const chunks = ( ( ( sourceSize + 1 ) + 511 ) >> 9 );
	BOOST_ASSERT( sourceSize + 1 >= chunks + 1 );

	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = chunks;
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	// call the partial sum kernel
	FindNonzeroIndicesInnerKernel<<< blocks, 256 >>>( deviceDestination, static_cast< boost::uint32_t* >( deviceWork ), deviceSource, sourceSize );

	if ( chunks > 1 ) {

		// recursively find partial sums of deviceDestination
		PartialSum( deviceDestination, static_cast< void* >( deviceDestination + chunks + 1 ), ( ( sourceSize + 1 ) - ( chunks + 1 ) ) * sizeof( boost::uint32_t ), chunks );

		// call the addition kernel
		PartialSumOuterKernel<<< blocks, 256 >>>( deviceDestination, static_cast< boost::uint32_t* >( deviceWork ), sourceSize );
	}

	boost::uint32_t destinationSize;
	cudaMemcpy(    // **TODO **FIXME: error checking
		&destinationSize,
		static_cast< boost::uint32_t* >( deviceWork ) + sourceSize,
		sizeof( boost::uint32_t ),
		cudaMemcpyDeviceToHost
	);

	// call the index kernel
	FindNonzeroIndicesOuterKernel<<< blocks, 256 >>>( deviceDestination, static_cast< boost::uint32_t* >( deviceWork ), deviceSource, sourceSize );

	return destinationSize;
}




//============================================================================
//    FindNonzeroIndices functions
//============================================================================


unsigned int BFindNonzeroIndices(
	boost::uint32_t* const deviceDestination,
	void* const deviceWork,
	bool const* const deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize
)
{
	return FindNonzeroIndicesHelper< bool >( deviceDestination, deviceWork, deviceSource, workSize, sourceSize );
}


unsigned int FFindNonzeroIndices(
	boost::uint32_t* const deviceDestination,
	void* const deviceWork,
	float const* const deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize
)
{
	return FindNonzeroIndicesHelper< float >( deviceDestination, deviceWork, deviceSource, workSize, sourceSize );
}


#ifdef CUDA_USE_DOUBLE

unsigned int DFindNonzeroIndices(
	boost::uint32_t* const deviceDestination,
	void* const deviceWork,
	double const* const deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize
)
{
	return FindNonzeroIndicesHelper< double >( deviceDestination, deviceWork, deviceSource, workSize, sourceSize );
}

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM
