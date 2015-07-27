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
	\file cuda_reduce.cu
	\brief CUDA reduction kernel
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
//    ReduceKernel kernel
//============================================================================


// adapted from the CUDA parallel reduction example
template< typename t_Type >
__global__ void ReduceKernel(
	t_Type* const destination,
	t_Type const* const source,
	unsigned int const destinationPitch,
	unsigned int const sourceSize,
	unsigned int const sourcePitch,
	unsigned int const copies
)
{
	__shared__ t_Type sums[ 256 ];

	unsigned int const gridSize   = ( gridDim.x  / copies );
	unsigned int const gridIndex  = ( blockIdx.x % copies );
	unsigned int const blockIndex = ( blockIdx.x / copies );

	unsigned int const blockOffset = gridIndex * sourcePitch;

	{	unsigned int const iiIndex  = ( blockIndex << 8 ) + threadIdx.x;
		unsigned int const iiStride = ( gridSize   << 8 );

		t_Type accumulator = 0;
		for ( unsigned int ii = iiIndex; ii < sourceSize; ii += iiStride )
			accumulator += source[ blockOffset + ii ];
		sums[ threadIdx.x ] = accumulator;
	}
	__syncthreads();

	// unrolled reduction
	if ( threadIdx.x < 128 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 128 ];
	__syncthreads();
	if ( threadIdx.x < 64 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 64 ];
	__syncthreads();
	if ( threadIdx.x < 32 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 32 ];
	__syncthreads();
	if ( threadIdx.x < 16 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 16 ];
	__syncthreads();
	if ( threadIdx.x < 8 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 8 ];
	__syncthreads();
	if ( threadIdx.x < 4 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 4 ];
	__syncthreads();
	if ( threadIdx.x < 2 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 2 ];
	__syncthreads();

	if ( threadIdx.x == 0 )
		destination[ gridIndex * destinationPitch + blockIndex ] = sums[ 0 ] + sums[ 1 ];
}




//============================================================================
//    ReduceHelper helper function
//============================================================================


template< typename t_Type >
t_Type const* ReduceHelper(
	void* deviceWorkVoid,
	t_Type* deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize,
	unsigned int const sourcePitch,
	unsigned int const copies
)
{
	t_Type* deviceWork = static_cast< t_Type* >( deviceWorkVoid );

	unsigned int size  = sourceSize;
	unsigned int pitch = sourcePitch;
	while ( pitch != 1 ) {

		/*
			start out with the maximum possible number of blocks (one unit of work
			per thread), and divide by an integer (so that each thread is doing
			the same amount of work) to get below the target
		*/
		unsigned int blocks = std::max( ( ( size + 255 ) >> 8 ), 1u );
		{	unsigned int const maximumBlocks = 65535u / copies;
			unsigned int const denominator = 1 + blocks / maximumBlocks;
			blocks = ( blocks + ( denominator - 1 ) ) / denominator;
			BOOST_ASSERT( ( blocks > 0 ) && ( blocks <= maximumBlocks ) );
		}

		// we need alighment, in order for reads to be coalesced, but *not* for the last reduction
		unsigned int const blockPitch = ( ( blocks > 1 ) ? ( ( blocks + 15 ) & ~15 ) : 1 );
		if ( workSize < ( blockPitch * copies ) * sizeof( t_Type ) )
			throw std::runtime_error( "Reduce: work buffer is too small!" );

		BOOST_ASSERT( pitch      >= size   );
		BOOST_ASSERT( blockPitch >= blocks );
		ReduceKernel< t_Type ><<< ( blocks * copies ), 256 >>>( deviceWork, deviceSource, blockPitch, size, pitch, copies );

		std::swap( deviceWork, deviceSource );
		size = blocks;
		pitch = blockPitch;
	}

	return deviceSource;
}




//============================================================================
//    Reduce functions
//============================================================================


boost::uint32_t const* UReduce(
	void* deviceWorkVoid,
	boost::uint32_t* deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize,
	unsigned int const sourcePitch,
	unsigned int const copies
)
{
	return ReduceHelper< boost::uint32_t >( deviceWorkVoid, deviceSource, workSize, sourceSize, sourcePitch, copies );
}


float const* FReduce(
	void* deviceWorkVoid,
	float* deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize,
	unsigned int const sourcePitch,
	unsigned int const copies
)
{
	return ReduceHelper< float >( deviceWorkVoid, deviceSource, workSize, sourceSize, sourcePitch, copies );
}


#ifdef CUDA_USE_DOUBLE

double const* DReduce(
	void* deviceWorkVoid,
	double* deviceSource,
	unsigned int const workSize,
	unsigned int const sourceSize,
	unsigned int const sourcePitch,
	unsigned int const copies
)
{
	return ReduceHelper< double >( deviceWorkVoid, deviceSource, workSize, sourceSize, sourcePitch, copies );
}

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM
