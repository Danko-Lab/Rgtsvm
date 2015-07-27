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
	\file cuda_sort.cu
	\brief CUDA sorting kernel
*/




#include "cuda_helpers.hpp"
#include "helpers.hpp"

#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

#include <utility>
#include <algorithm>
#include <stdexcept>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    FindLargestKernel kernel
//============================================================================


// adapted from the CUDA bitonic sort example
template< typename t_KeyType, typename t_ValueType >
__global__ void FindLargestKernel(
	t_KeyType* const destinationKeys,
	t_ValueType* const destinationValues,
	unsigned int const resultSize,
	unsigned int const logSortSize,
	t_KeyType const* const sourceKeys,
	t_ValueType const* const sourceValues,
	unsigned int const sourceSize
)
{
	__shared__ t_KeyType   keysCache[   512 ];
	__shared__ t_ValueType valuesCache[ 512 ];

	unsigned int const sortSize = ( 1u << logSortSize );

	for ( unsigned int ii = blockIdx.x; ( ii << 9 ) < sourceSize; ii += gridDim.x ) {

		unsigned int blockSize = ( ( ( ii << 9 ) + 512 < sourceSize ) ? 512 : ( sourceSize - ( ii << 9 ) ) );
		if ( threadIdx.x < blockSize ) {

			unsigned int const index = ( ii << 9 ) + threadIdx.x;
			keysCache[   threadIdx.x ] = sourceKeys[   index ];
			valuesCache[ threadIdx.x ] = sourceValues[ index ];
		}
		if ( 256 + threadIdx.x < blockSize ) {

			unsigned int const index = ( ii << 9 ) + 256 + threadIdx.x;
			keysCache[   256 + threadIdx.x ] = sourceKeys[   index ];
			valuesCache[ 256 + threadIdx.x ] = sourceValues[ index ];
		}
		__syncthreads();

		for ( unsigned int jj = 2; jj <= sortSize; jj += jj ) {

			unsigned int kk = ( jj >> 1 );
			{	unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				if ( index1 < blockSize ) {

					unsigned int const index2 = ( index1 & ~( kk - 1 ) ) - ( index1 & ( kk - 1 ) ) - 1;
					t_KeyType const key1 = keysCache[ index1 ];
					t_KeyType const key2 = keysCache[ index2 ];
					if ( key2 < key1 ) {

						keysCache[ index1 ] = key2;
						keysCache[ index2 ] = key1;

						t_ValueType const value1 = valuesCache[ index1 ];
						valuesCache[ index1 ] = valuesCache[ index2 ];
						valuesCache[ index2 ] = value1;
					}
				}
			}
			kk >>= 1;
			__syncthreads();

			for ( ; kk > 0; kk >>= 1 ) {

				unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				if ( index1 < blockSize ) {

					unsigned int const index2 = ( index1 ^ kk );
					t_KeyType const key1 = keysCache[ index1 ];
					t_KeyType const key2 = keysCache[ index2 ];
					if ( key2 < key1 ) {

						keysCache[ index1 ] = key2;
						keysCache[ index2 ] = key1;

						t_ValueType const value1 = valuesCache[ index1 ];
						valuesCache[ index1 ] = valuesCache[ index2 ];
						valuesCache[ index2 ] = value1;
					}
				}
				__syncthreads();
			}
		}

		unsigned int const blockDestinationSize = ( blockSize >> logSortSize ) * resultSize + ( ( ( blockSize & ( sortSize - 1 ) ) < resultSize ) ? ( blockSize & ( sortSize - 1 ) ) : resultSize  );
		if ( threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * resultSize + threadIdx.x;
			unsigned int const sourceIndex = ( ( threadIdx.x / resultSize ) << logSortSize ) + ( threadIdx.x % resultSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		if ( 256 + threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * resultSize + 256 + threadIdx.x;
			unsigned int const sourceIndex = ( ( ( 256 + threadIdx.x ) / resultSize ) << logSortSize ) + ( ( 256 + threadIdx.x ) % resultSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		__syncthreads();
	}
}




//============================================================================
//    FindLargestHelper helper function
//============================================================================


template< typename t_KeyType, typename t_ValueType >
std::pair< std::pair< t_KeyType const*, t_ValueType const* >, unsigned int > FindLargestHelper(
	void* deviceWork1,
	void* deviceWork2,
	t_KeyType* deviceSourceKeys,
	t_ValueType* deviceSourceValues,
	unsigned int const workSize,
	unsigned int const resultSize,
	unsigned int const destinationSize,
	unsigned int const sourceSize
)
{
	t_KeyType*   deviceDestinationKeys   = reinterpret_cast< t_KeyType*   >( deviceWork1 );
	t_ValueType* deviceDestinationValues = reinterpret_cast< t_ValueType* >( deviceWork2 );

	if ( ( resultSize < 1 ) || ( resultSize > 256 ) )
		throw std::runtime_error( "FindLargest: result size must be between 1 and 256" );
	if ( destinationSize < resultSize )
		throw std::runtime_error( "FindLargest: destination size must be at least as large as result size" );

	unsigned int logResultSize = 0;
	if ( resultSize <= 16 ) {

		if ( resultSize <= 4 ) {

			if ( resultSize <= 2 ) {

				if ( resultSize <= 1 )
					logResultSize = 0;
				else
					logResultSize = 1;
			}
			else
				logResultSize = 2;
		}
		else {

			if ( resultSize <= 8 )
				logResultSize = 3;
			else
				logResultSize = 4;
		}
	}
	else {

		if ( resultSize <= 64 ) {

			if ( resultSize <= 32 )
				logResultSize = 5;
			else
				logResultSize = 6;
		}
		else {

			if ( resultSize <= 128 )
				logResultSize = 7;
			else
				logResultSize = 8;
		}
	}
	BOOST_ASSERT( resultSize <= ( 1u << logResultSize ) );
	BOOST_ASSERT( resultSize > ( ( 1u << logResultSize ) >> 1 ) );

	unsigned int const logSortSize = std::min( logResultSize + 2, 9u );

	unsigned int size = sourceSize;
	while ( size > destinationSize ) {

		unsigned int const nextSize = ( ( size >> logSortSize ) * resultSize ) + std::min( size & ( ( 1u << logSortSize ) - 1 ), resultSize );
		if ( workSize < nextSize * std::max( sizeof( t_KeyType ), sizeof( t_ValueType ) ) )
			throw std::runtime_error( "FindLargest: work buffer is too small!" );

		/*
			start out with the maximum possible number of blocks (one unit of work
			per thread), and divide by an integer (so that each thread is doing
			the same amount of work) to get below the target
		*/
		unsigned int blocks = ( ( size + 511 ) >> 9 );
		{	unsigned int const maximumBlocks = 65535u;
			unsigned int const denominator = 1 + blocks / maximumBlocks;
			blocks = ( blocks + ( denominator - 1 ) ) / denominator;
			BOOST_ASSERT( blocks <= maximumBlocks );
		}

		FindLargestKernel< t_KeyType, t_ValueType ><<< blocks, 256 >>>(
			deviceDestinationKeys,
			deviceDestinationValues,
			resultSize,
			logSortSize,
			deviceSourceKeys,
			deviceSourceValues,
			size
		);

		std::swap( deviceDestinationKeys,   deviceSourceKeys   );
		std::swap( deviceDestinationValues, deviceSourceValues );
		size = nextSize;
	}

	return std::pair< std::pair< t_KeyType const*, t_ValueType const* >, unsigned int >(
		std::pair< t_KeyType const*, t_ValueType const* >( deviceSourceKeys, deviceSourceValues ),
		size
	);
}




//============================================================================
//    FindLargest functions
//============================================================================


std::pair< std::pair< boost::uint32_t const*, boost::uint32_t const* >, unsigned int > UUFindLargest(
	void* deviceWork1,
	void* deviceWork2,
	boost::uint32_t* deviceSourceKeys,
	boost::uint32_t* deviceSourceValues,
	unsigned int const workSize,
	unsigned int const resultSize,
	unsigned int const destinationSize,
	unsigned int const sourceSize
)
{
	return FindLargestHelper( deviceWork1, deviceWork2, deviceSourceKeys, deviceSourceValues, workSize, resultSize, destinationSize, sourceSize );
}


std::pair< std::pair< float const*, boost::uint32_t const* >, unsigned int > FUFindLargest(
	void* deviceWork1,
	void* deviceWork2,
	float* deviceSourceKeys,
	boost::uint32_t* deviceSourceValues,
	unsigned int const workSize,
	unsigned int const resultSize,
	unsigned int const destinationSize,
	unsigned int const sourceSize
)
{
	return FindLargestHelper( deviceWork1, deviceWork2, deviceSourceKeys, deviceSourceValues, workSize, resultSize, destinationSize, sourceSize );
}


#ifdef CUDA_USE_DOUBLE

std::pair< std::pair< double const*, boost::uint32_t const* >, unsigned int > DUFindLargest(
	void* deviceWork1,
	void* deviceWork2,
	double* deviceSourceKeys,
	boost::uint32_t* deviceSourceValues,
	unsigned int const workSize,
	unsigned int const resultSize,
	unsigned int const destinationSize,
	unsigned int const sourceSize
)
{
	return FindLargestHelper( deviceWork1, deviceWork2, deviceSourceKeys, deviceSourceValues, workSize, resultSize, destinationSize, sourceSize );
}

#endif    // CUDA_USE_DOUBLE




}    // namespace CUDA




}    // namespace GTSVM
