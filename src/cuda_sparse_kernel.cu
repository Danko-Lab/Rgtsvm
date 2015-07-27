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
	\file cuda_sparse_kernel.cu
	\brief CUDA kernel functions
*/




#include "cuda_sparse_kernel.hpp"
#include "cuda_helpers.hpp"
#include "cuda_reduce.hpp"
#include "cuda_find_largest.hpp"
#include "cuda_partial_sum.hpp"
#include "cuda_array.hpp"
#include "gtsvm.h"
#include "helpers.hpp"

#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

#include <set>
#include <stdexcept>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    CUDA_INFINITY and CUDA_NEGATIVE_INFINITY macros
//============================================================================


#define CUDA_INFINITY          ( __int_as_float( 0x7f800000 ) )
#define CUDA_NEGATIVE_INFINITY ( __int_as_float( 0xff800000 ) )




//============================================================================
//    Kernel functors
//============================================================================


template< int t_Kernel >
struct Kernel { };


template<>
struct Kernel< GTSVM_KERNEL_GAUSSIAN > {

	static __device__ CUDA_FLOAT_DOUBLE Calculate(
		float innerProduct,
		float normSquared1,
		float normSquared2,
		float kernelParameter1,
		float kernelParameter2,
		float kernelParameter3
	)
	{
		return exp( kernelParameter1 * ( 2 * innerProduct - normSquared1 - normSquared2 ) );
	}
};


template<>
struct Kernel< GTSVM_KERNEL_POLYNOMIAL > {

	static __device__ CUDA_FLOAT_DOUBLE Calculate(
		float innerProduct,
		float normSquared1,
		float normSquared2,
		float kernelParameter1,
		float kernelParameter2,
		float kernelParameter3
	)
	{
		return pow( kernelParameter1 * innerProduct + kernelParameter2, kernelParameter3 );
	}
};


template<>
struct Kernel< GTSVM_KERNEL_SIGMOID > {

	static __device__ CUDA_FLOAT_DOUBLE Calculate(
		float innerProduct,
		float normSquared1,
		float normSquared2,
		float kernelParameter1,
		float kernelParameter2,
		float kernelParameter3
	)
	{
		CUDA_FLOAT_DOUBLE const exponent = exp( 2 * ( kernelParameter1 * innerProduct + kernelParameter2 ) );
		return( ( exponent - 1 ) / ( exponent + 1 ) );
	}
};




//============================================================================
//    SparseEvaluateKernelKernel16 kernel
//============================================================================


template< int t_Kernel >
__global__ void SparseEvaluateKernelKernel16(
	CUDA_FLOAT_DOUBLE* const destination,
	float const* const batchVectorsTranspose,
	float const* const batchVectorNormsSquared,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const clusters,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3
)
{
	__shared__ CUDA::SparseKernelClusterHeader clusterHeader;
	__shared__ boost::uint32_t nonzeroIndicesCache[ 256 ];
	__shared__ float batchVectorsCache[ 256 ];
	__shared__ float trainingVectorsCache[ 256 ];
	__shared__ CUDA_FLOAT_DOUBLE sums[ 256 ];

	unsigned int const index = ( threadIdx.x & 15 );

	CUDA_FLOAT_DOUBLE sum = 0;

	for ( unsigned int ii = blockIdx.x; ii < clusters; ii += gridDim.x ) {

		switch( threadIdx.x ) {

			case 0: clusterHeader.size                     = clusterHeaders[ ii ].size;                     break;
			case 1: clusterHeader.nonzeros                 = clusterHeaders[ ii ].nonzeros;                 break;
			//case 2: clusterHeader.responses                = clusterHeaders[ ii ].responses;                break;
			//case 3: clusterHeader.labels                   = clusterHeaders[ ii ].labels;                   break;
			case 4: clusterHeader.alphas                   = clusterHeaders[ ii ].alphas;                   break;
			case 5: clusterHeader.nonzeroIndices           = clusterHeaders[ ii ].nonzeroIndices;           break;
			case 6: clusterHeader.vectorsTranspose         = clusterHeaders[ ii ].vectorsTranspose;         break;
			case 8: clusterHeader.vectorNormsSquared       = clusterHeaders[ ii ].vectorNormsSquared;       break;
			//case 9: clusterHeader.vectorKernelNormsSquared = clusterHeaders[ ii ].vectorKernelNormsSquared; break;
			default: break;
		}
		__syncthreads();

		float accumulator = 0;

		for ( unsigned int jj = 0; jj < clusterHeader.nonzeros; jj += 16 ) {

			if ( ( jj & 255 ) == 0 ) {

				boost::uint32_t value = 0;
				unsigned int const offset = jj + threadIdx.x;
				if ( offset < clusterHeader.nonzeros )
					value = clusterHeader.nonzeroIndices[ offset ];
				nonzeroIndicesCache[ threadIdx.x ] = value;
			}
			__syncthreads();

			{	float batchValue    = 0;
				float trainingValue = 0;
				unsigned int const offset = jj + ( threadIdx.x >> 4 );
				if ( offset < clusterHeader.nonzeros ) {

					batchValue = batchVectorsTranspose[ nonzeroIndicesCache[ offset & 255 ] * 16 + index ];
					trainingValue = clusterHeader.vectorsTranspose[ offset * 16 + index ];
				}
				batchVectorsCache[    threadIdx.x ] = batchValue;
				trainingVectorsCache[ threadIdx.x ] = trainingValue;
			}
			__syncthreads();

			if ( index < clusterHeader.size ) {

				#pragma unroll
				for ( unsigned int kk = 0; kk < 256; kk += 16 )
					accumulator += trainingVectorsCache[ kk + ( threadIdx.x & 15 ) ] * batchVectorsCache[ kk + ( threadIdx.x >> 4 ) ];
			}
		}
		__syncthreads();

		if ( index < clusterHeader.size ) {

			float const alpha = clusterHeader.alphas[ index ];
			float const batchVectorNormSquared = batchVectorNormsSquared[ threadIdx.x >> 4 ];
			float const trainingVectorNormSquared = clusterHeader.vectorNormsSquared[ index ];

			sum += alpha * Kernel< t_Kernel >::Calculate( accumulator, trainingVectorNormSquared, batchVectorNormSquared, kernelParameter1, kernelParameter2, kernelParameter3 );
		}
		__syncthreads();
	}

	sums[ threadIdx.x ] = sum;
	__syncthreads();

	if ( ( threadIdx.x & 15 ) < 8 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 8 ];
	__syncthreads();
	if ( ( threadIdx.x & 15 ) < 4 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 4 ];
	__syncthreads();
	if ( ( threadIdx.x & 15 ) < 2 )
		sums[ threadIdx.x ] += sums[ threadIdx.x + 2 ];
	__syncthreads();

	if ( ( threadIdx.x & 15 ) == 0 ) {

		unsigned int const destinationStride = ( ( gridDim.x + 15 ) & ~15 );
		unsigned int destinationIndex = blockIdx.x + ( threadIdx.x >> 4 ) * destinationStride;

		destination[ destinationIndex ] = sums[ threadIdx.x ] + sums[ threadIdx.x + 1 ];
	}
}




//============================================================================
//    SparseEvaluateKernelKernel256 kernel
//============================================================================


template< int t_Kernel >
__global__ void SparseEvaluateKernelKernel256(
	CUDA_FLOAT_DOUBLE* const destination,
	float const* const batchVectorsTranspose,
	float const* const batchVectorNormsSquared,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const clusters,
	unsigned int const classes,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3
)
{
	__shared__ CUDA::SparseKernelClusterHeader clusterHeader;
	__shared__ boost::uint32_t nonzeroIndicesCache[ 256 ];
	__shared__ float batchVectorsCache[ 256 ];

	for ( unsigned int ii = blockIdx.x; ii < clusters; ii += gridDim.x ) {

		switch( threadIdx.x ) {

			case 0: clusterHeader.size                     = clusterHeaders[ ii ].size;                     break;
			case 1: clusterHeader.nonzeros                 = clusterHeaders[ ii ].nonzeros;                 break;
			//case 2: clusterHeader.responses                = clusterHeaders[ ii ].responses;                break;
			//case 3: clusterHeader.labels                   = clusterHeaders[ ii ].labels;                   break;
			case 4: clusterHeader.alphas                   = clusterHeaders[ ii ].alphas;                   break;
			case 5: clusterHeader.nonzeroIndices           = clusterHeaders[ ii ].nonzeroIndices;           break;
			case 6: clusterHeader.vectorsTranspose         = clusterHeaders[ ii ].vectorsTranspose;         break;
			case 8: clusterHeader.vectorNormsSquared       = clusterHeaders[ ii ].vectorNormsSquared;       break;
			//case 9: clusterHeader.vectorKernelNormsSquared = clusterHeaders[ ii ].vectorKernelNormsSquared; break;
			default: break;
		}
		__syncthreads();

		float accumulator0 = 0;
		float accumulator1 = 0;
		float accumulator2 = 0;
		float accumulator3 = 0;
		float accumulator4 = 0;
		float accumulator5 = 0;
		float accumulator6 = 0;
		float accumulator7 = 0;
		float accumulator8 = 0;
		float accumulator9 = 0;
		float accumulatorA = 0;
		float accumulatorB = 0;
		float accumulatorC = 0;
		float accumulatorD = 0;
		float accumulatorE = 0;
		float accumulatorF = 0;

		for ( unsigned int jj = 0; jj < clusterHeader.nonzeros; jj += 16 ) {

			if ( ( jj & 255 ) == 0 ) {

				boost::uint32_t value = 0;
				unsigned int const offset = jj + threadIdx.x;
				if ( offset < clusterHeader.nonzeros )
					value = clusterHeader.nonzeroIndices[ offset ];
				nonzeroIndicesCache[ threadIdx.x ] = value;
			}
			__syncthreads();

			{	float batchValue = 0;
				unsigned int const offset = jj + ( threadIdx.x >> 4 );
				if ( offset < clusterHeader.nonzeros )
					batchValue = batchVectorsTranspose[ nonzeroIndicesCache[ offset & 255 ] * 16 + ( threadIdx.x & 15 ) ];
				batchVectorsCache[ ( ( threadIdx.x & 15 ) << 4 ) + ( threadIdx.x >> 4 ) ] = batchValue;
			}
			__syncthreads();

			if ( threadIdx.x < clusterHeader.size ) {

				unsigned int const kkMax = ( ( jj + 16 <= clusterHeader.nonzeros ) ? 16 : ( clusterHeader.nonzeros - jj ) );
				#pragma unroll
				for ( unsigned int kk = 0; kk < kkMax; ++kk ) {

					float const trainingValue = clusterHeader.vectorsTranspose[ ( jj + kk ) * 256 + threadIdx.x ];
					accumulator0 += trainingValue * batchVectorsCache[  0 * 16 + kk ];
					accumulator1 += trainingValue * batchVectorsCache[  1 * 16 + kk ];
					accumulator2 += trainingValue * batchVectorsCache[  2 * 16 + kk ];
					accumulator3 += trainingValue * batchVectorsCache[  3 * 16 + kk ];
					accumulator4 += trainingValue * batchVectorsCache[  4 * 16 + kk ];
					accumulator5 += trainingValue * batchVectorsCache[  5 * 16 + kk ];
					accumulator6 += trainingValue * batchVectorsCache[  6 * 16 + kk ];
					accumulator7 += trainingValue * batchVectorsCache[  7 * 16 + kk ];
					accumulator8 += trainingValue * batchVectorsCache[  8 * 16 + kk ];
					accumulator9 += trainingValue * batchVectorsCache[  9 * 16 + kk ];
					accumulatorA += trainingValue * batchVectorsCache[ 10 * 16 + kk ];
					accumulatorB += trainingValue * batchVectorsCache[ 11 * 16 + kk ];
					accumulatorC += trainingValue * batchVectorsCache[ 12 * 16 + kk ];
					accumulatorD += trainingValue * batchVectorsCache[ 13 * 16 + kk ];
					accumulatorE += trainingValue * batchVectorsCache[ 14 * 16 + kk ];
					accumulatorF += trainingValue * batchVectorsCache[ 15 * 16 + kk ];
				}
			}
		}
		__syncthreads();

		if ( threadIdx.x < 16 )
			batchVectorsCache[ threadIdx.x ] = batchVectorNormsSquared[ threadIdx.x ];
		__syncthreads();

		{	unsigned int const destinationStride = clusters * 256;
			unsigned int destinationIndex = ii * 256 + threadIdx.x;

			if ( threadIdx.x < clusterHeader.size ) {

				float const trainingVectorNormSquared = clusterHeader.vectorNormsSquared[ threadIdx.x ];
				for ( unsigned int jj = 0; jj < classes; ++jj ) {

					float const alpha = clusterHeader.alphas[ threadIdx.x + jj * 256 ];

					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator0, trainingVectorNormSquared, batchVectorsCache[  0 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator1, trainingVectorNormSquared, batchVectorsCache[  1 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator2, trainingVectorNormSquared, batchVectorsCache[  2 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator3, trainingVectorNormSquared, batchVectorsCache[  3 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator4, trainingVectorNormSquared, batchVectorsCache[  4 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator5, trainingVectorNormSquared, batchVectorsCache[  5 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator6, trainingVectorNormSquared, batchVectorsCache[  6 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator7, trainingVectorNormSquared, batchVectorsCache[  7 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator8, trainingVectorNormSquared, batchVectorsCache[  8 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulator9, trainingVectorNormSquared, batchVectorsCache[  9 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulatorA, trainingVectorNormSquared, batchVectorsCache[ 10 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulatorB, trainingVectorNormSquared, batchVectorsCache[ 11 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulatorC, trainingVectorNormSquared, batchVectorsCache[ 12 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulatorD, trainingVectorNormSquared, batchVectorsCache[ 13 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulatorE, trainingVectorNormSquared, batchVectorsCache[ 14 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
					destination[ destinationIndex ] = alpha * Kernel< t_Kernel >::Calculate( accumulatorF, trainingVectorNormSquared, batchVectorsCache[ 15 ], kernelParameter1, kernelParameter2, kernelParameter3 ); destinationIndex += destinationStride;
				}
			}
			else {

				for ( unsigned int jj = 0; jj < classes; ++jj ) {

					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
					destination[ destinationIndex ] = 0; destinationIndex += destinationStride;
				}
			}
		}
		__syncthreads();
	}
}




//============================================================================
//    SparseUpdateKernelKernel16 kernel
//============================================================================


template< int t_Kernel >
__global__ void SparseUpdateKernelKernel16(
	float const* const batchVectorsTranspose,
	float const* const batchVectorNormsSquared,
	float const* const batchDeltaAlphas,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const clusters,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3
)
{
	__shared__ CUDA::SparseKernelClusterHeader clusterHeader;
	__shared__ boost::uint32_t nonzeroIndicesCache[ 256 ];
	__shared__ float batchVectorsCache[ 256 ];
	__shared__ float trainingVectorsCache[ 256 ];
	__shared__ CUDA_FLOAT_DOUBLE sums[ 256 ];

	unsigned int const index = ( threadIdx.x & 15 );

	for ( unsigned int ii = blockIdx.x; ii < clusters; ii += gridDim.x ) {

		switch( threadIdx.x ) {

			case 0: clusterHeader.size                     = clusterHeaders[ ii ].size;                     break;
			case 1: clusterHeader.nonzeros                 = clusterHeaders[ ii ].nonzeros;                 break;
			case 2: clusterHeader.responses                = clusterHeaders[ ii ].responses;                break;
			//case 3: clusterHeader.labels                   = clusterHeaders[ ii ].labels;                   break;
			//case 4: clusterHeader.alphas                   = clusterHeaders[ ii ].alphas;                   break;
			case 5: clusterHeader.nonzeroIndices           = clusterHeaders[ ii ].nonzeroIndices;           break;
			case 6: clusterHeader.vectorsTranspose         = clusterHeaders[ ii ].vectorsTranspose;         break;
			case 8: clusterHeader.vectorNormsSquared       = clusterHeaders[ ii ].vectorNormsSquared;       break;
			//case 9: clusterHeader.vectorKernelNormsSquared = clusterHeaders[ ii ].vectorKernelNormsSquared; break;
			default: break;
		}
		__syncthreads();

		float accumulator = 0;

		for ( unsigned int jj = 0; jj < clusterHeader.nonzeros; jj += 16 ) {

			if ( ( jj & 255 ) == 0 ) {

				boost::uint32_t value = 0;
				unsigned int const offset = jj + threadIdx.x;
				if ( offset < clusterHeader.nonzeros )
					value = clusterHeader.nonzeroIndices[ offset ];
				nonzeroIndicesCache[ threadIdx.x ] = value;
			}
			__syncthreads();

			{	float batchValue    = 0;
				float trainingValue = 0;
				unsigned int const offset = jj + ( threadIdx.x >> 4 );
				if ( offset < clusterHeader.nonzeros ) {

					batchValue = batchVectorsTranspose[ nonzeroIndicesCache[ offset & 255 ] * 16 + index ];
					trainingValue = clusterHeader.vectorsTranspose[ offset * 16 + index ];
				}
				batchVectorsCache[    threadIdx.x ] = batchValue;
				trainingVectorsCache[ threadIdx.x ] = trainingValue;
			}
			__syncthreads();

			if ( index < clusterHeader.size ) {

				#pragma unroll
				for ( unsigned int kk = 0; kk < 256; kk += 16 )
					accumulator += trainingVectorsCache[ kk + ( threadIdx.x & 15 ) ] * batchVectorsCache[ kk + ( threadIdx.x >> 4 ) ];
			}
		}
		__syncthreads();

		CUDA_FLOAT_DOUBLE sum = 0;

		if ( index < clusterHeader.size ) {

			float const deltaAlpha = batchDeltaAlphas[ threadIdx.x >> 4 ];
			float const batchVectorNormSquared = batchVectorNormsSquared[ threadIdx.x >> 4 ];
			float const trainingVectorNormSquared = clusterHeader.vectorNormsSquared[ index ];

			sum = deltaAlpha * Kernel< t_Kernel >::Calculate( accumulator, trainingVectorNormSquared, batchVectorNormSquared, kernelParameter1, kernelParameter2, kernelParameter3 );
		}

		sums[ ( threadIdx.x >> 4 ) | ( ( threadIdx.x & 15 ) << 4 ) ] = sum;
		__syncthreads();

		if ( ( threadIdx.x & 15 ) < 8 )
			sums[ threadIdx.x ] += sums[ threadIdx.x + 8 ];
		__syncthreads();
		if ( ( threadIdx.x & 15 ) < 4 )
			sums[ threadIdx.x ] += sums[ threadIdx.x + 4 ];
		__syncthreads();
		if ( ( threadIdx.x & 15 ) < 2 )
			sums[ threadIdx.x ] += sums[ threadIdx.x + 2 ];
		__syncthreads();

		if ( ( ( threadIdx.x & 15 ) == 0 ) && ( ( threadIdx.x >> 4 ) < clusterHeader.size ) )
			clusterHeader.responses[ threadIdx.x >> 4 ] += sums[ threadIdx.x ] + sums[ threadIdx.x + 1 ];
	}
}




//============================================================================
//    SparseUpdateKernelKernel256 kernel
//============================================================================


template< int t_Kernel >
__global__ void SparseUpdateKernelKernel256(
	float const* const batchVectorsTranspose,
	float const* const batchVectorNormsSquared,
	float const* const batchDeltaAlphas,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const clusters,
	unsigned int const classes,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3
)
{
	__shared__ CUDA::SparseKernelClusterHeader clusterHeader;
	__shared__ boost::uint32_t nonzeroIndicesCache[ 256 ];
	__shared__ float batchVectorsCache[ 256 ];

	for ( unsigned int ii = blockIdx.x; ii < clusters; ii += gridDim.x ) {

		switch( threadIdx.x ) {

			case 0: clusterHeader.size                     = clusterHeaders[ ii ].size;                     break;
			case 1: clusterHeader.nonzeros                 = clusterHeaders[ ii ].nonzeros;                 break;
			case 2: clusterHeader.responses                = clusterHeaders[ ii ].responses;                break;
			//case 3: clusterHeader.labels                   = clusterHeaders[ ii ].labels;                   break;
			//case 4: clusterHeader.alphas                   = clusterHeaders[ ii ].alphas;                   break;
			case 5: clusterHeader.nonzeroIndices           = clusterHeaders[ ii ].nonzeroIndices;           break;
			case 6: clusterHeader.vectorsTranspose         = clusterHeaders[ ii ].vectorsTranspose;         break;
			case 8: clusterHeader.vectorNormsSquared       = clusterHeaders[ ii ].vectorNormsSquared;       break;
			//case 9: clusterHeader.vectorKernelNormsSquared = clusterHeaders[ ii ].vectorKernelNormsSquared; break;
			default: break;
		}
		__syncthreads();

		float accumulator0 = 0;
		float accumulator1 = 0;
		float accumulator2 = 0;
		float accumulator3 = 0;
		float accumulator4 = 0;
		float accumulator5 = 0;
		float accumulator6 = 0;
		float accumulator7 = 0;
		float accumulator8 = 0;
		float accumulator9 = 0;
		float accumulatorA = 0;
		float accumulatorB = 0;
		float accumulatorC = 0;
		float accumulatorD = 0;
		float accumulatorE = 0;
		float accumulatorF = 0;

		for ( unsigned int jj = 0; jj < clusterHeader.nonzeros; jj += 16 ) {

			if ( ( jj & 255 ) == 0 ) {

				boost::uint32_t value = 0;
				unsigned int const offset = jj + threadIdx.x;
				if ( offset < clusterHeader.nonzeros )
					value = clusterHeader.nonzeroIndices[ offset ];
				nonzeroIndicesCache[ threadIdx.x ] = value;
			}
			__syncthreads();

			{	float batchValue = 0;
				unsigned int const offset = jj + ( threadIdx.x >> 4 );
				if ( offset < clusterHeader.nonzeros )
					batchValue = batchVectorsTranspose[ nonzeroIndicesCache[ offset & 255 ] * 16 + ( threadIdx.x & 15 ) ];
				batchVectorsCache[ ( ( threadIdx.x & 15 ) << 4 ) + ( threadIdx.x >> 4 ) ] = batchValue;
			}
			__syncthreads();

			if ( threadIdx.x < clusterHeader.size ) {

				unsigned int const kkMax = ( ( jj + 16 <= clusterHeader.nonzeros ) ? 16 : ( clusterHeader.nonzeros - jj ) );
				#pragma unroll
				for ( unsigned int kk = 0; kk < kkMax; ++kk ) {

					float const trainingValue = clusterHeader.vectorsTranspose[ ( jj + kk ) * 256 + threadIdx.x ];
					accumulator0 += trainingValue * batchVectorsCache[  0 * 16 + kk ];
					accumulator1 += trainingValue * batchVectorsCache[  1 * 16 + kk ];
					accumulator2 += trainingValue * batchVectorsCache[  2 * 16 + kk ];
					accumulator3 += trainingValue * batchVectorsCache[  3 * 16 + kk ];
					accumulator4 += trainingValue * batchVectorsCache[  4 * 16 + kk ];
					accumulator5 += trainingValue * batchVectorsCache[  5 * 16 + kk ];
					accumulator6 += trainingValue * batchVectorsCache[  6 * 16 + kk ];
					accumulator7 += trainingValue * batchVectorsCache[  7 * 16 + kk ];
					accumulator8 += trainingValue * batchVectorsCache[  8 * 16 + kk ];
					accumulator9 += trainingValue * batchVectorsCache[  9 * 16 + kk ];
					accumulatorA += trainingValue * batchVectorsCache[ 10 * 16 + kk ];
					accumulatorB += trainingValue * batchVectorsCache[ 11 * 16 + kk ];
					accumulatorC += trainingValue * batchVectorsCache[ 12 * 16 + kk ];
					accumulatorD += trainingValue * batchVectorsCache[ 13 * 16 + kk ];
					accumulatorE += trainingValue * batchVectorsCache[ 14 * 16 + kk ];
					accumulatorF += trainingValue * batchVectorsCache[ 15 * 16 + kk ];
				}
			}
		}
		__syncthreads();

		if ( threadIdx.x < 16 )
			batchVectorsCache[ threadIdx.x ] = batchVectorNormsSquared[ threadIdx.x ];
		__syncthreads();

		if ( threadIdx.x < clusterHeader.size ) {

			float trainingVectorNormSquared = clusterHeader.vectorNormsSquared[ threadIdx.x ];

			accumulator0 = Kernel< t_Kernel >::Calculate( accumulator0, trainingVectorNormSquared, batchVectorsCache[  0 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator1 = Kernel< t_Kernel >::Calculate( accumulator1, trainingVectorNormSquared, batchVectorsCache[  1 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator2 = Kernel< t_Kernel >::Calculate( accumulator2, trainingVectorNormSquared, batchVectorsCache[  2 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator3 = Kernel< t_Kernel >::Calculate( accumulator3, trainingVectorNormSquared, batchVectorsCache[  3 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator4 = Kernel< t_Kernel >::Calculate( accumulator4, trainingVectorNormSquared, batchVectorsCache[  4 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator5 = Kernel< t_Kernel >::Calculate( accumulator5, trainingVectorNormSquared, batchVectorsCache[  5 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator6 = Kernel< t_Kernel >::Calculate( accumulator6, trainingVectorNormSquared, batchVectorsCache[  6 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator7 = Kernel< t_Kernel >::Calculate( accumulator7, trainingVectorNormSquared, batchVectorsCache[  7 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator8 = Kernel< t_Kernel >::Calculate( accumulator8, trainingVectorNormSquared, batchVectorsCache[  8 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulator9 = Kernel< t_Kernel >::Calculate( accumulator9, trainingVectorNormSquared, batchVectorsCache[  9 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulatorA = Kernel< t_Kernel >::Calculate( accumulatorA, trainingVectorNormSquared, batchVectorsCache[ 10 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulatorB = Kernel< t_Kernel >::Calculate( accumulatorB, trainingVectorNormSquared, batchVectorsCache[ 11 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulatorC = Kernel< t_Kernel >::Calculate( accumulatorC, trainingVectorNormSquared, batchVectorsCache[ 12 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulatorD = Kernel< t_Kernel >::Calculate( accumulatorD, trainingVectorNormSquared, batchVectorsCache[ 13 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulatorE = Kernel< t_Kernel >::Calculate( accumulatorE, trainingVectorNormSquared, batchVectorsCache[ 14 ], kernelParameter1, kernelParameter2, kernelParameter3 );
			accumulatorF = Kernel< t_Kernel >::Calculate( accumulatorF, trainingVectorNormSquared, batchVectorsCache[ 15 ], kernelParameter1, kernelParameter2, kernelParameter3 );
		}

		__syncthreads();

		for ( unsigned int jj = 0; jj < classes; jj += 16 ) {

			{	float batchValue = 0;
				unsigned int const offset = jj + ( threadIdx.x >> 4 );
				if ( offset < classes )
					batchValue = batchDeltaAlphas[ offset * 16 + ( threadIdx.x & 15 ) ];
				batchVectorsCache[ threadIdx.x ] = batchValue;
			}
			__syncthreads();

			if ( threadIdx.x < clusterHeader.size ) {

				unsigned int const kkMax = ( ( jj + 16 <= classes ) ? 16 : ( classes - jj ) );
				#pragma unroll
				for ( unsigned int kk = 0; kk < kkMax; ++kk ) {

					CUDA_FLOAT_DOUBLE sum = 0;
					sum += batchVectorsCache[  0 + kk * 16 ] * accumulator0;
					sum += batchVectorsCache[  1 + kk * 16 ] * accumulator1;
					sum += batchVectorsCache[  2 + kk * 16 ] * accumulator2;
					sum += batchVectorsCache[  3 + kk * 16 ] * accumulator3;
					sum += batchVectorsCache[  4 + kk * 16 ] * accumulator4;
					sum += batchVectorsCache[  5 + kk * 16 ] * accumulator5;
					sum += batchVectorsCache[  6 + kk * 16 ] * accumulator6;
					sum += batchVectorsCache[  7 + kk * 16 ] * accumulator7;
					sum += batchVectorsCache[  8 + kk * 16 ] * accumulator8;
					sum += batchVectorsCache[  9 + kk * 16 ] * accumulator9;
					sum += batchVectorsCache[ 10 + kk * 16 ] * accumulatorA;
					sum += batchVectorsCache[ 11 + kk * 16 ] * accumulatorB;
					sum += batchVectorsCache[ 12 + kk * 16 ] * accumulatorC;
					sum += batchVectorsCache[ 13 + kk * 16 ] * accumulatorD;
					sum += batchVectorsCache[ 14 + kk * 16 ] * accumulatorE;
					sum += batchVectorsCache[ 15 + kk * 16 ] * accumulatorF;
					clusterHeader.responses[ threadIdx.x + ( jj + kk ) * 256 ] += sum;
				}
			}

			__syncthreads();
		}
	}
}




//============================================================================
//    SparseKernelArrayUpdateKernel kernel
//============================================================================


__global__ void SparseKernelArrayUpdateKernel(
	float* const batchAlphas,
	boost::uint32_t const* const batchIndices,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const classes,
	unsigned int const logMaximumClusterSize
)
{
	unsigned int const cluster = ( batchIndices[ threadIdx.x ] >> logMaximumClusterSize );
	unsigned int const index = ( batchIndices[ threadIdx.x ] & ( ( 1u << logMaximumClusterSize ) - 1 ) );

	float* const trainingAlphas = clusterHeaders[ cluster ].alphas;

	for ( unsigned int ii = 0; ii < classes; ++ii ) {

		float const oldAlpha = trainingAlphas[ index + ( ii << logMaximumClusterSize ) ];
		float const newAlpha = batchAlphas[ threadIdx.x + ii * 16 ];
		trainingAlphas[ index + ( ii << logMaximumClusterSize ) ] = newAlpha;
		batchAlphas[ threadIdx.x + ii * 16 ] = newAlpha - oldAlpha;
	}
}




//============================================================================
//    SparseCalculateBiasKernel kernel
//============================================================================


__global__ void SparseCalculateBiasKernel(
	CUDA_FLOAT_DOUBLE* const numeratorDestination,
	boost::uint32_t* const denominatorDestination,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	float const regularization
)
{
	__shared__ CUDA_FLOAT_DOUBLE numeratorSums[ 256 ];
	__shared__ boost::uint32_t denominatorSums[ 256 ];

	CUDA_FLOAT_DOUBLE numerator = 0;
	boost::uint32_t denominator = 0;

	for ( unsigned int ii = ( blockIdx.x << ( 8 - logMaximumClusterSize ) ); ii < clusters; ii += ( gridDim.x << ( 8 - logMaximumClusterSize ) ) ) {

		unsigned int const cluster = ii + ( threadIdx.x >> logMaximumClusterSize );
		unsigned int const index = ( threadIdx.x & ( ( 1u << logMaximumClusterSize ) - 1 ) );

		if ( cluster < clusters ) {

			if ( index < clusterHeaders[ cluster ].size ) {

				CUDA_FLOAT_DOUBLE const response = clusterHeaders[ cluster ].responses[ index ];
				boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
				float const alpha = fabs( clusterHeaders[ cluster ].alphas[ index ] );

				if ( ( alpha > 0 ) && ( alpha < regularization ) ) {

					numerator += ( ( label > 0 ) ? 1 : -1 ) - response;
					++denominator;
				}
			}
		}
	}

	numeratorSums[   threadIdx.x ] = numerator;
	denominatorSums[ threadIdx.x ] = denominator;
	__syncthreads();

	// unrolled reduction
	if ( threadIdx.x < 128 ) {

		numeratorSums[   threadIdx.x ] += numeratorSums[   threadIdx.x + 128 ];
		denominatorSums[ threadIdx.x ] += denominatorSums[ threadIdx.x + 128 ];
	}
	__syncthreads();
	if ( threadIdx.x < 64 ) {

		numeratorSums[   threadIdx.x ] += numeratorSums[   threadIdx.x + 64 ];
		denominatorSums[ threadIdx.x ] += denominatorSums[ threadIdx.x + 64 ];
	}
	__syncthreads();
	if ( threadIdx.x < 32 ) {

		numeratorSums[   threadIdx.x ] += numeratorSums[   threadIdx.x + 32 ];
		denominatorSums[ threadIdx.x ] += denominatorSums[ threadIdx.x + 32 ];
	}
	__syncthreads();
	if ( threadIdx.x < 16 ) {

		numeratorSums[   threadIdx.x ] += numeratorSums[   threadIdx.x + 16 ];
		denominatorSums[ threadIdx.x ] += denominatorSums[ threadIdx.x + 16 ];
	}
	__syncthreads();
	if ( threadIdx.x < 8 ) {

		numeratorSums[   threadIdx.x ] += numeratorSums[   threadIdx.x + 8 ];
		denominatorSums[ threadIdx.x ] += denominatorSums[ threadIdx.x + 8 ];
	}
	__syncthreads();
	if ( threadIdx.x < 4 ) {

		numeratorSums[   threadIdx.x ] += numeratorSums[   threadIdx.x + 4 ];
		denominatorSums[ threadIdx.x ] += denominatorSums[ threadIdx.x + 4 ];
	}
	__syncthreads();
	if ( threadIdx.x < 2 ) {

		numeratorSums[   threadIdx.x ] += numeratorSums[   threadIdx.x + 2 ];
		denominatorSums[ threadIdx.x ] += denominatorSums[ threadIdx.x + 2 ];
	}
	__syncthreads();

	if ( threadIdx.x == 0 ) {

		numeratorDestination[   blockIdx.x ] = numeratorSums[   0 ] + numeratorSums[   1 ];
		denominatorDestination[ blockIdx.x ] = denominatorSums[ 0 ] + denominatorSums[ 1 ];
	}
}




//============================================================================
//    SparseCalculateObjectivesKernel kernel
//============================================================================


__global__ void SparseCalculateObjectivesKernel(
	CUDA_FLOAT_DOUBLE* const primalDestination,
	CUDA_FLOAT_DOUBLE* const dualDestination,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const classes,
	float const regularization,
	float const bias
)
{
	__shared__ CUDA_FLOAT_DOUBLE primalSums[ 256 ];
	__shared__ CUDA_FLOAT_DOUBLE dualSums[   256 ];

	CUDA_FLOAT_DOUBLE primalSum = 0;
	CUDA_FLOAT_DOUBLE dualSum = 0;

	for ( unsigned int ii = ( blockIdx.x << ( 8 - logMaximumClusterSize ) ); ii < clusters; ii += ( gridDim.x << ( 8 - logMaximumClusterSize ) ) ) {

		unsigned int const cluster = ii + ( threadIdx.x >> logMaximumClusterSize );
		unsigned int const index = ( threadIdx.x & ( ( 1u << logMaximumClusterSize ) - 1 ) );

		if ( cluster < clusters ) {

			if ( index < clusterHeaders[ cluster ].size ) {

				if ( classes == 1 ) {

					CUDA_FLOAT_DOUBLE const response = clusterHeaders[ cluster ].responses[ index ];
					boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
					float const alpha = clusterHeaders[ cluster ].alphas[ index ];

					CUDA_FLOAT_DOUBLE hinge;
					if ( label > 0 )
						hinge = 1 - ( response + bias );
					else
						hinge = 1 + ( response + bias );
					if ( hinge < 0 )
						hinge = 0;

					CUDA_FLOAT_DOUBLE const weight = 0.5 * alpha * response;

					primalSum += weight + regularization * hinge;
					dualSum += std::fabs( alpha ) - weight;
				}
				else {

					boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
					CUDA_FLOAT_DOUBLE const* pResponse = &clusterHeaders[ cluster ].responses[ index ];
					float const* pAlpha = &clusterHeaders[ cluster ].alphas[ index ];

					CUDA_FLOAT_DOUBLE hinge  = CUDA_NEGATIVE_INFINITY;
					CUDA_FLOAT_DOUBLE weight = 0;
					CUDA_FLOAT_DOUBLE hingeShift;
					float trueAlpha;

					for ( unsigned int jj = 0; jj < classes; ++jj, pResponse += ( 1u << logMaximumClusterSize ), pAlpha += ( 1u << logMaximumClusterSize ) ) {

						CUDA_FLOAT_DOUBLE const response = *pResponse;
						float const alpha = *pAlpha;

						weight += alpha * response;

						float innerHinge = response;
						if ( jj == label ) {

							trueAlpha = alpha;
							hingeShift = 1 - response;
							innerHinge -= 1;
						}
						if ( innerHinge > hinge )
							hinge = innerHinge;
					}
					hinge += hingeShift;
					weight *= 0.5;

					primalSum += weight + regularization * hinge;
					dualSum += trueAlpha - weight;
				}
			}
		}
	}

	primalSums[ threadIdx.x ] = primalSum;
	dualSums[   threadIdx.x ] = dualSum;
	__syncthreads();

	// unrolled reduction
	if ( threadIdx.x < 128 ) {

		primalSums[ threadIdx.x ] += primalSums[ threadIdx.x + 128 ];
		dualSums[   threadIdx.x ] += dualSums[   threadIdx.x + 128 ];
	}
	__syncthreads();
	if ( threadIdx.x < 64 ) {

		primalSums[ threadIdx.x ] += primalSums[ threadIdx.x + 64 ];
		dualSums[   threadIdx.x ] += dualSums[   threadIdx.x + 64 ];
	}
	__syncthreads();
	if ( threadIdx.x < 32 ) {

		primalSums[ threadIdx.x ] += primalSums[ threadIdx.x + 32 ];
		dualSums[   threadIdx.x ] += dualSums[   threadIdx.x + 32 ];
	}
	__syncthreads();
	if ( threadIdx.x < 16 ) {

		primalSums[ threadIdx.x ] += primalSums[ threadIdx.x + 16 ];
		dualSums[   threadIdx.x ] += dualSums[   threadIdx.x + 16 ];
	}
	__syncthreads();
	if ( threadIdx.x < 8 ) {

		primalSums[ threadIdx.x ] += primalSums[ threadIdx.x + 8 ];
		dualSums[   threadIdx.x ] += dualSums[   threadIdx.x + 8 ];
	}
	__syncthreads();
	if ( threadIdx.x < 4 ) {

		primalSums[ threadIdx.x ] += primalSums[ threadIdx.x + 4 ];
		dualSums[   threadIdx.x ] += dualSums[   threadIdx.x + 4 ];
	}
	__syncthreads();
	if ( threadIdx.x < 2 ) {

		primalSums[ threadIdx.x ] += primalSums[ threadIdx.x + 2 ];
		dualSums[   threadIdx.x ] += dualSums[   threadIdx.x + 2 ];
	}
	__syncthreads();

	if ( threadIdx.x == 0 ) {

		primalDestination[ blockIdx.x ] = primalSums[ 0 ] + primalSums[ 1 ];
		dualDestination[   blockIdx.x ] = dualSums[   0 ] + dualSums[   1 ];
	}
}




//============================================================================
//    SparseKernelFindLargestScoreKernel kernel
//============================================================================


__global__ void SparseKernelFindLargestScoreKernel(
	float* destinationKeys,
	boost::uint32_t* destinationValues,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const classes,
	unsigned int const destinationSize,
	unsigned int const logSortSize,
	float const regularization
)
{
	__shared__ float    keysCache[   512 ];
	__shared__ boost::uint32_t valuesCache[ 512 ];

	unsigned int const sortSize = ( 1u << logSortSize );

	for ( unsigned int ii = blockIdx.x; ( ii << ( 9 - logMaximumClusterSize ) ) < clusters; ii += gridDim.x ) {

		for ( unsigned int jj = 0; jj < 2; ++jj ) {

			unsigned int const cluster = ( ( ( ii << 9 ) + ( jj << 8 ) + threadIdx.x ) >> logMaximumClusterSize );
			unsigned int const index = ( threadIdx.x & ( ( 1u << logMaximumClusterSize ) - 1 ) );

			float score = CUDA_NEGATIVE_INFINITY;
			unsigned int destinationIndex = static_cast< unsigned int >( -1 );

			if ( cluster < clusters ) {

				if ( index < clusterHeaders[ cluster ].size ) {

					destinationIndex = ( cluster << logMaximumClusterSize ) + index;

#ifndef SECOND_ORDER
					if ( classes == 1 ) {

						CUDA_FLOAT_DOUBLE const response = clusterHeaders[ cluster ].responses[ index ];
						boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
						float const alpha = fabs( clusterHeaders[ cluster ].alphas[ index ] );

						float gradient = 0;
						if ( label > 0 )
							gradient = 1 - response;
						else
							gradient = 1 + response;

						score = fabs( gradient );
						if (
							( ( gradient > 0 ) && ( ! ( alpha < regularization ) ) ) ||
							( ( gradient < 0 ) && ( ! ( alpha >              0 ) ) )
						)
						{
							score = -score;
						}
					}
					else {

						boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
						CUDA_FLOAT_DOUBLE const* pResponse = &clusterHeaders[ cluster ].responses[ index ];
						float const* pAlpha = &clusterHeaders[ cluster ].alphas[ index ];

						CUDA_FLOAT_DOUBLE maximumGradient = CUDA_NEGATIVE_INFINITY;
						CUDA_FLOAT_DOUBLE minimumGradient = CUDA_INFINITY;

						for ( unsigned int kk = 0; kk < classes; ++kk, pResponse += ( 1u << logMaximumClusterSize ), pAlpha += ( 1u << logMaximumClusterSize ) ) {

							CUDA_FLOAT_DOUBLE const response = *pResponse;
							float const alpha = *pAlpha;

							float gradient = -response;
							float bound = 0;
							if ( kk == label ) {

								gradient += 1;
								bound = regularization;
							}

							if ( ( alpha < bound ) && ( gradient > maximumGradient ) )
								maximumGradient = gradient;
							if ( gradient < minimumGradient )
								minimumGradient = gradient;
						}

						score = maximumGradient - minimumGradient;
					}
#else    // SECOND_ORDER
					if ( classes == 1 ) {

						CUDA_FLOAT_DOUBLE const response = clusterHeaders[ cluster ].responses[ index ];
						boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
						float const alpha = fabs( clusterHeaders[ cluster ].alphas[ index ] );
						float const scale = clusterHeaders[ cluster ].vectorKernelNormsSquared[ index ];

						float gradient = 0;
						if ( label > 0 )
							gradient = 1 - response;
						else
							gradient = 1 + response;

						float newAlpha = alpha + gradient / scale;
						if ( newAlpha > regularization )
							newAlpha = regularization;
						else if ( newAlpha < 0 )
							newAlpha = 0;

						float delta = newAlpha - alpha;
						score = ( gradient - 0.5 * delta * scale ) * delta;
					}
					else {

						boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
						float const scale = clusterHeaders[ cluster ].vectorKernelNormsSquared[ index ];
						CUDA_FLOAT_DOUBLE const* pResponse = &clusterHeaders[ cluster ].responses[ index ];
						float const* pAlpha = &clusterHeaders[ cluster ].alphas[ index ];

						CUDA_FLOAT_DOUBLE minimumGradient = CUDA_INFINITY;

						for ( unsigned int kk = 0; kk < classes; ++kk, pResponse += ( 1u << logMaximumClusterSize ) ) {

							CUDA_FLOAT_DOUBLE const response = *pResponse;

							float gradient = -response;
							if ( kk == label )
								gradient += 1;

							if ( gradient < minimumGradient )
								minimumGradient = gradient;
						}
						pResponse -= ( classes << logMaximumClusterSize );

						for ( unsigned int kk = 0; kk < classes; ++kk, pResponse += ( 1u << logMaximumClusterSize ), pAlpha += ( 1u << logMaximumClusterSize ) ) {

							CUDA_FLOAT_DOUBLE const response = *pResponse;
							float const alpha = *pAlpha;

							float gradient = -response;
							float bound = 0;
							if ( kk == label ) {

								gradient += 1;
								bound = regularization;
							}

							float delta = 0.5 * ( gradient - minimumGradient ) / scale;
							if ( delta > 0 ) {

								if ( delta > bound - alpha )
									delta = bound - alpha;

								float const newScore = ( ( gradient - minimumGradient ) - delta * scale ) * delta;
								if ( newScore > score )
									score = newScore;
							}
						}
					}
#endif    // SECOND_ORDER
				}
			}

			keysCache[   ( jj << 8 ) + threadIdx.x ] = score;
			valuesCache[ ( jj << 8 ) + threadIdx.x ] = destinationIndex;
		}
		__syncthreads();

		for ( unsigned int jj = 2; jj <= sortSize; jj += jj ) {

			unsigned int kk = ( jj >> 1 );
			{	unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				unsigned int const index2 = ( index1 & ~( kk - 1 ) ) - ( index1 & ( kk - 1 ) ) - 1;
				float const key1 = keysCache[ index1 ];
				float const key2 = keysCache[ index2 ];
				if ( key2 < key1 ) {

					keysCache[ index1 ] = key2;
					keysCache[ index2 ] = key1;

					boost::uint32_t const value1 = valuesCache[ index1 ];
					valuesCache[ index1 ] = valuesCache[ index2 ];
					valuesCache[ index2 ] = value1;
				}
			}
			kk >>= 1;
			__syncthreads();

			for ( ; kk > 0; kk >>= 1 ) {

				unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				unsigned int const index2 = ( index1 ^ kk );
				float const key1 = keysCache[ index1 ];
				float const key2 = keysCache[ index2 ];
				if ( key2 < key1 ) {

					keysCache[ index1 ] = key2;
					keysCache[ index2 ] = key1;

					boost::uint32_t const value1 = valuesCache[ index1 ];
					valuesCache[ index1 ] = valuesCache[ index2 ];
					valuesCache[ index2 ] = value1;
				}
				__syncthreads();
			}
		}

		unsigned int const blockDestinationSize = ( 512 >> logSortSize ) * destinationSize;
		if ( threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * destinationSize + threadIdx.x;
			unsigned int const sourceIndex = ( ( threadIdx.x / destinationSize ) << logSortSize ) + ( threadIdx.x % destinationSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		if ( 256 + threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * destinationSize + 256 + threadIdx.x;
			unsigned int const sourceIndex = ( ( ( 256 + threadIdx.x ) / destinationSize ) << logSortSize ) + ( ( 256 + threadIdx.x ) % destinationSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		__syncthreads();
	}
}




//============================================================================
//    SparseKernelFindLargestPositiveGradientKernel kernel
//============================================================================


__global__ void SparseKernelFindLargestPositiveGradientKernel(
	float* destinationKeys,
	boost::uint32_t* destinationValues,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const destinationSize,
	unsigned int const logSortSize,
	float const regularization
)
{
	__shared__ float    keysCache[   512 ];
	__shared__ boost::uint32_t valuesCache[ 512 ];

	unsigned int const sortSize = ( 1u << logSortSize );

	for ( unsigned int ii = blockIdx.x; ( ii << ( 9 - logMaximumClusterSize ) ) < clusters; ii += gridDim.x ) {

		for ( unsigned int jj = 0; jj < 2; ++jj ) {

			unsigned int const cluster = ( ( ( ii << 9 ) + ( jj << 8 ) + threadIdx.x ) >> logMaximumClusterSize );
			unsigned int const index = ( threadIdx.x & ( ( 1u << logMaximumClusterSize ) - 1 ) );

			float score = CUDA_NEGATIVE_INFINITY;
			unsigned int destinationIndex = static_cast< unsigned int >( -1 );

			if ( cluster < clusters ) {

				if ( index < clusterHeaders[ cluster ].size ) {

					CUDA_FLOAT_DOUBLE const response = clusterHeaders[ cluster ].responses[ index ];
					boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
					float const alpha = clusterHeaders[ cluster ].alphas[ index ];

					if ( label > 0 ) {

						float const gradient = 1 - response;
						if ( alpha < regularization )
							score = gradient;
					}
					else {

						float const gradient = -1 - response;
						if ( alpha < -0 )
							score = gradient;
					}

					destinationIndex = ( cluster << logMaximumClusterSize ) + index;
				}
			}

			keysCache[   ( jj << 8 ) + threadIdx.x ] = score;
			valuesCache[ ( jj << 8 ) + threadIdx.x ] = destinationIndex;
		}
		__syncthreads();

		for ( unsigned int jj = 2; jj <= sortSize; jj += jj ) {

			unsigned int kk = ( jj >> 1 );
			{	unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				unsigned int const index2 = ( index1 & ~( kk - 1 ) ) - ( index1 & ( kk - 1 ) ) - 1;
				float const key1 = keysCache[ index1 ];
				float const key2 = keysCache[ index2 ];
				if ( key2 < key1 ) {

					keysCache[ index1 ] = key2;
					keysCache[ index2 ] = key1;

					boost::uint32_t const value1 = valuesCache[ index1 ];
					valuesCache[ index1 ] = valuesCache[ index2 ];
					valuesCache[ index2 ] = value1;
				}
			}
			kk >>= 1;
			__syncthreads();

			for ( ; kk > 0; kk >>= 1 ) {

				unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				unsigned int const index2 = ( index1 ^ kk );
				float const key1 = keysCache[ index1 ];
				float const key2 = keysCache[ index2 ];
				if ( key2 < key1 ) {

					keysCache[ index1 ] = key2;
					keysCache[ index2 ] = key1;

					boost::uint32_t const value1 = valuesCache[ index1 ];
					valuesCache[ index1 ] = valuesCache[ index2 ];
					valuesCache[ index2 ] = value1;
				}
				__syncthreads();
			}
		}

		unsigned int const blockDestinationSize = ( 512 >> logSortSize ) * destinationSize;
		if ( threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * destinationSize + threadIdx.x;
			unsigned int const sourceIndex = ( ( threadIdx.x / destinationSize ) << logSortSize ) + ( threadIdx.x % destinationSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		if ( 256 + threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * destinationSize + 256 + threadIdx.x;
			unsigned int const sourceIndex = ( ( ( 256 + threadIdx.x ) / destinationSize ) << logSortSize ) + ( ( 256 + threadIdx.x ) % destinationSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		__syncthreads();
	}
}




//============================================================================
//    SparseKernelFindLargestNegativeGradientKernel kernel
//============================================================================


__global__ void SparseKernelFindLargestNegativeGradientKernel(
	float* destinationKeys,
	boost::uint32_t* destinationValues,
	CUDA::SparseKernelClusterHeader const* const clusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const destinationSize,
	unsigned int const logSortSize,
	float const regularization
)
{
	__shared__ float    keysCache[   512 ];
	__shared__ boost::uint32_t valuesCache[ 512 ];

	unsigned int const sortSize = ( 1u << logSortSize );

	for ( unsigned int ii = blockIdx.x; ( ii << ( 9 - logMaximumClusterSize ) ) < clusters; ii += gridDim.x ) {

		for ( unsigned int jj = 0; jj < 2; ++jj ) {

			unsigned int const cluster = ( ( ( ii << 9 ) + ( jj << 8 ) + threadIdx.x ) >> logMaximumClusterSize );
			unsigned int const index = ( threadIdx.x & ( ( 1u << logMaximumClusterSize ) - 1 ) );

			float score = CUDA_NEGATIVE_INFINITY;
			unsigned int destinationIndex = static_cast< unsigned int >( -1 );

			if ( cluster < clusters ) {

				if ( index < clusterHeaders[ cluster ].size ) {

					CUDA_FLOAT_DOUBLE const response = clusterHeaders[ cluster ].responses[ index ];
					boost::int32_t const label = clusterHeaders[ cluster ].labels[ index ];
					float const alpha = clusterHeaders[ cluster ].alphas[ index ];

					if ( label > 0 ) {

						float const gradient = 1 - response;
						if ( alpha > 0 )
							score = -gradient;
					}
					else {

						float const gradient = -1 - response;
						if ( alpha > -regularization )
							score = -gradient;
					}

					destinationIndex = ( cluster << logMaximumClusterSize ) + index;
				}
			}

			keysCache[   ( jj << 8 ) + threadIdx.x ] = score;
			valuesCache[ ( jj << 8 ) + threadIdx.x ] = destinationIndex;
		}
		__syncthreads();

		for ( unsigned int jj = 2; jj <= sortSize; jj += jj ) {

			unsigned int kk = ( jj >> 1 );
			{	unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				unsigned int const index2 = ( index1 & ~( kk - 1 ) ) - ( index1 & ( kk - 1 ) ) - 1;
				float const key1 = keysCache[ index1 ];
				float const key2 = keysCache[ index2 ];
				if ( key2 < key1 ) {

					keysCache[ index1 ] = key2;
					keysCache[ index2 ] = key1;

					boost::uint32_t const value1 = valuesCache[ index1 ];
					valuesCache[ index1 ] = valuesCache[ index2 ];
					valuesCache[ index2 ] = value1;
				}
			}
			kk >>= 1;
			__syncthreads();

			for ( ; kk > 0; kk >>= 1 ) {

				unsigned int const index1 = ( ( threadIdx.x & ~( kk - 1 ) ) << 1 ) | kk | ( threadIdx.x & ( kk - 1 ) );
				unsigned int const index2 = ( index1 ^ kk );
				float const key1 = keysCache[ index1 ];
				float const key2 = keysCache[ index2 ];
				if ( key2 < key1 ) {

					keysCache[ index1 ] = key2;
					keysCache[ index2 ] = key1;

					boost::uint32_t const value1 = valuesCache[ index1 ];
					valuesCache[ index1 ] = valuesCache[ index2 ];
					valuesCache[ index2 ] = value1;
				}
				__syncthreads();
			}
		}

		unsigned int const blockDestinationSize = ( 512 >> logSortSize ) * destinationSize;
		if ( threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * destinationSize + threadIdx.x;
			unsigned int const sourceIndex = ( ( threadIdx.x / destinationSize ) << logSortSize ) + ( threadIdx.x % destinationSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		if ( 256 + threadIdx.x < blockDestinationSize ) {

			unsigned int const destinationIndex = ( ii << ( 9 - logSortSize ) ) * destinationSize + 256 + threadIdx.x;
			unsigned int const sourceIndex = ( ( ( 256 + threadIdx.x ) / destinationSize ) << logSortSize ) + ( ( 256 + threadIdx.x ) % destinationSize );
			destinationKeys[   destinationIndex ] = keysCache[   sourceIndex ];
			destinationValues[ destinationIndex ] = valuesCache[ sourceIndex ];
		}
		__syncthreads();
	}
}




//============================================================================
//    SparseEvaluateKernel function
//============================================================================


CUDA_FLOAT_DOUBLE const* SparseEvaluateKernel(
	void* deviceWork1,
	void* deviceWork2,
	float const* const deviceBatchVectorsTranspose,
	float const* const deviceBatchVectorNormsSquared,
	SparseKernelClusterHeader const* const deviceClusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const classes,
	unsigned int const workSize,
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3
)
{
	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = clusters;
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	CUDA_FLOAT_DOUBLE const* result = NULL;

	if ( logMaximumClusterSize == 4 ) {

		if ( classes != 1 )
			throw std::runtime_error( "SparseEvaluateKernel: multiclass only implemented for size-256 clusters" );

		if ( workSize < ( clusters << 8 ) * sizeof( CUDA_FLOAT_DOUBLE ) )
			throw std::runtime_error( "SparseEvaluateKernel: work buffer is too small!" );

		// call the kernel
		switch( kernel ) {

			case GTSVM_KERNEL_GAUSSIAN: {

				SparseEvaluateKernelKernel16< GTSVM_KERNEL_GAUSSIAN ><<< blocks, 256 >>>(
					static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceClusterHeaders,
					clusters,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_POLYNOMIAL: {

				SparseEvaluateKernelKernel16< GTSVM_KERNEL_POLYNOMIAL ><<< blocks, 256 >>>(
					static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceClusterHeaders,
					clusters,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_SIGMOID: {

				SparseEvaluateKernelKernel16< GTSVM_KERNEL_SIGMOID ><<< blocks, 256 >>>(
					static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceClusterHeaders,
					clusters,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			default: throw std::runtime_error( "SparseEvaluateKernel: unknown kernel" );
		}

#ifdef CUDA_USE_DOUBLE
		result = DReduce(
			deviceWork2,
			static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
			workSize,
			blocks,
			( ( blocks + 15 ) & ~15 ),
			16
		);
#else    // CUDA_USE_DOUBLE
		result = FReduce(
			deviceWork2,
			static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
			workSize,
			blocks,
			( ( blocks + 15 ) & ~15 ),
			16
		);
#endif    // CUDA_USE_DOUBLE
	}
	else if ( logMaximumClusterSize == 8 ) {

		if ( workSize < ( ( clusters * classes ) << 12 ) * sizeof( CUDA_FLOAT_DOUBLE ) )
			throw std::runtime_error( "SparseEvaluateKernel: work buffer is too small!" );

		// call the kernel
		switch( kernel ) {

			case GTSVM_KERNEL_GAUSSIAN: {

				SparseEvaluateKernelKernel256< GTSVM_KERNEL_GAUSSIAN ><<< blocks, 256 >>>(
					static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceClusterHeaders,
					clusters,
					classes,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_POLYNOMIAL: {

				SparseEvaluateKernelKernel256< GTSVM_KERNEL_POLYNOMIAL ><<< blocks, 256 >>>(
					static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceClusterHeaders,
					clusters,
					classes,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_SIGMOID: {

				SparseEvaluateKernelKernel256< GTSVM_KERNEL_SIGMOID ><<< blocks, 256 >>>(
					static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceClusterHeaders,
					clusters,
					classes,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			default: throw std::runtime_error( "SparseEvaluateKernel: unknown kernel" );
		}

#ifdef CUDA_USE_DOUBLE
		result = DReduce(
			deviceWork2,
			static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
			workSize,
			clusters * 256,
			clusters * 256,
			16 * classes
		);
#else    // CUDA_USE_DOUBLE
		result = FReduce(
			deviceWork2,
			static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
			workSize,
			clusters * 256,
			clusters * 256,
			16 * classes
		);
#endif    // CUDA_USE_DOUBLE
	}
	else
		throw std::runtime_error( "SparseEvaluateKernel: maximum cluster size must be 16 or 256!" );

	return result;
}




//============================================================================
//    SparseUpdateKernel function
//============================================================================


void SparseUpdateKernel(
	float const* const deviceBatchVectorsTranspose,
	float const* const deviceBatchVectorNormsSquared,
	float* const deviceBatchAlphas,
	boost::uint32_t const* const deviceBatchIndices,
	SparseKernelClusterHeader const* const deviceClusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const classes,
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3
)
{
	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = clusters;
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	// update trainingAlphas, and put the change in the alphas into deviceBatchAlphas
	SparseKernelArrayUpdateKernel<<< 1, 16 >>>(
		deviceBatchAlphas,
		deviceBatchIndices,
		deviceClusterHeaders,
		classes,
		logMaximumClusterSize
	);

	if ( logMaximumClusterSize == 4 ) {

		if ( classes != 1 )
			throw std::runtime_error( "SparseUpdateKernel: multiclass only implemented for size-256 clusters" );

		// call the kernel
		switch( kernel ) {

			case GTSVM_KERNEL_GAUSSIAN: {

				SparseUpdateKernelKernel16< GTSVM_KERNEL_GAUSSIAN ><<< blocks, 256 >>>(
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceBatchAlphas,
					deviceClusterHeaders,
					clusters,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_POLYNOMIAL: {

				SparseUpdateKernelKernel16< GTSVM_KERNEL_POLYNOMIAL ><<< blocks, 256 >>>(
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceBatchAlphas,
					deviceClusterHeaders,
					clusters,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_SIGMOID: {

				SparseUpdateKernelKernel16< GTSVM_KERNEL_SIGMOID ><<< blocks, 256 >>>(
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceBatchAlphas,
					deviceClusterHeaders,
					clusters,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			default: throw std::runtime_error( "SparseUpdateKernel: unknown kernel" );
		}
	}
	else if ( logMaximumClusterSize == 8 ) {

		// call the kernel
		switch( kernel ) {

			case GTSVM_KERNEL_GAUSSIAN: {

				SparseUpdateKernelKernel256< GTSVM_KERNEL_GAUSSIAN ><<< blocks, 256 >>>(
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceBatchAlphas,
					deviceClusterHeaders,
					clusters,
					classes,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_POLYNOMIAL: {

				SparseUpdateKernelKernel256< GTSVM_KERNEL_POLYNOMIAL ><<< blocks, 256 >>>(
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceBatchAlphas,
					deviceClusterHeaders,
					clusters,
					classes,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			case GTSVM_KERNEL_SIGMOID: {

				SparseUpdateKernelKernel256< GTSVM_KERNEL_SIGMOID ><<< blocks, 256 >>>(
					deviceBatchVectorsTranspose,
					deviceBatchVectorNormsSquared,
					deviceBatchAlphas,
					deviceClusterHeaders,
					clusters,
					classes,
					kernelParameter1,
					kernelParameter2,
					kernelParameter3
				);
				break;
			}

			default: throw std::runtime_error( "SparseUpdateKernel: unknown kernel" );
		}
	}
	else
		throw std::runtime_error( "SparseUpdateKernel: maximum cluster size must be 16 or 256!" );
}




//============================================================================
//    SparseCalculateBias function
//============================================================================


std::pair< CUDA_FLOAT_DOUBLE const*, boost::uint32_t const* > SparseCalculateBias(
	void* deviceWork1,
	void* deviceWork2,
	void* deviceWork3,
	void* deviceWork4,
	SparseKernelClusterHeader const* const deviceClusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const workSize,
	float const regularization
)
{
	unsigned int const size = ( ( ( clusters << logMaximumClusterSize ) + 255 ) >> 8 );
	if ( workSize < size * std::max( sizeof( CUDA_FLOAT_DOUBLE ), sizeof( boost::uint32_t ) ) )
		throw std::runtime_error( "SparseCalculateBias: work buffer is too small!" );

	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = size;
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	SparseCalculateBiasKernel<<< blocks, 256 >>>(
		reinterpret_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
		reinterpret_cast< boost::uint32_t* >( deviceWork3 ),
		deviceClusterHeaders,
		logMaximumClusterSize,
		clusters,
		regularization
	);

#ifdef CUDA_USE_DOUBLE
	CUDA_FLOAT_DOUBLE const* pNumerator = DReduce(
		deviceWork2,
		static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
		workSize,
		blocks,
		blocks,
		1
	);
#else    // CUDA_USE_DOUBLE
	CUDA_FLOAT_DOUBLE const* pNumerator = FReduce(
		deviceWork2,
		static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
		workSize,
		blocks,
		blocks,
		1
	);
#endif    // CUDA_USE_DOUBLE

	boost::uint32_t const* pDenominator = UReduce(
		deviceWork4,
		static_cast< boost::uint32_t* >( deviceWork3 ),
		workSize,
		blocks,
		blocks,
		1
	);

	return std::pair< CUDA_FLOAT_DOUBLE const*, boost::uint32_t const* >( pNumerator, pDenominator );
}




//============================================================================
//    SparseCalculateObjectives function
//============================================================================


std::pair< CUDA_FLOAT_DOUBLE const*, CUDA_FLOAT_DOUBLE const* > SparseCalculateObjectives(
	void* deviceWork1,
	void* deviceWork2,
	void* deviceWork3,
	void* deviceWork4,
	SparseKernelClusterHeader const* const deviceClusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const classes,
	unsigned int const workSize,
	float const regularization,
	float const bias
)
{
	unsigned int const size = ( ( ( clusters << logMaximumClusterSize ) + 255 ) >> 8 );
	if ( workSize < size * sizeof( CUDA_FLOAT_DOUBLE ) )
		throw std::runtime_error( "SparseCalculateObjectives: work buffer is too small!" );

	/*
		start out with the maximum possible number of blocks (one unit of work
		per thread), and divide by an integer (so that each thread is doing
		the same amount of work) to get below the target
	*/
	unsigned int blocks = size;
	{	unsigned int const maximumBlocks = 65535u;
		unsigned int const denominator = 1 + blocks / maximumBlocks;
		blocks = ( blocks + ( denominator - 1 ) ) / denominator;
		BOOST_ASSERT( blocks <= maximumBlocks );
	}

	SparseCalculateObjectivesKernel<<< blocks, 256 >>>(
		reinterpret_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
		reinterpret_cast< CUDA_FLOAT_DOUBLE* >( deviceWork3 ),
		deviceClusterHeaders,
		logMaximumClusterSize,
		clusters,
		classes,
		regularization,
		bias
	);

#ifdef CUDA_USE_DOUBLE
	CUDA_FLOAT_DOUBLE const* pPrimal = DReduce(
		deviceWork2,
		static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
		workSize,
		blocks,
		blocks,
		1
	);
#else    // CUDA_USE_DOUBLE
	CUDA_FLOAT_DOUBLE const* pPrimal = FReduce(
		deviceWork2,
		static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork1 ),
		workSize,
		blocks,
		blocks,
		1
	);
#endif    // CUDA_USE_DOUBLE

#ifdef CUDA_USE_DOUBLE
	CUDA_FLOAT_DOUBLE const* pDual = DReduce(
		deviceWork4,
		static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork3 ),
		workSize,
		blocks,
		blocks,
		1
	);
#else    // CUDA_USE_DOUBLE
	CUDA_FLOAT_DOUBLE const* pDual = FReduce(
		deviceWork4,
		static_cast< CUDA_FLOAT_DOUBLE* >( deviceWork3 ),
		workSize,
		blocks,
		blocks,
		1
	);
#endif    // CUDA_USE_DOUBLE

	return std::pair< CUDA_FLOAT_DOUBLE const*, CUDA_FLOAT_DOUBLE const* >( pPrimal, pDual );
}




//============================================================================
//    SparseKernelFindLargestScore function
//============================================================================


void SparseKernelFindLargestScore(
	float* const destinationKeys,
	boost::uint32_t* const destinationValues,
	void* deviceWork1,
	void* deviceWork2,
	void* deviceWork3,
	void* deviceWork4,
	SparseKernelClusterHeader const* const deviceClusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const classes,
	unsigned int const workSize,
	unsigned int const resultSize,
	unsigned int const destinationSize,
	float const regularization
)
{
	if ( ( resultSize < 1 ) || ( resultSize > 256 ) )
		throw std::runtime_error( "SparseKernelFindLargestScore: result size must be between 1 and 256" );

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

	{	unsigned int const clusterBlocks = ( ( ( clusters << logMaximumClusterSize ) + 511 ) >> 9 );
		unsigned int const size = ( clusterBlocks << ( 9 - logSortSize ) ) * resultSize;
		if ( workSize < size * std::max( sizeof( float ), sizeof( boost::uint32_t ) ) )
			throw std::runtime_error( "SparseKernelFindLargestScore: work buffer is too small!" );

		/*
			start out with the maximum possible number of blocks (one unit of work
			per thread), and divide by an integer (so that each thread is doing
			the same amount of work) to get below the target
		*/
		unsigned int blocks = clusterBlocks;
		{	unsigned int const maximumBlocks = 65535u;
			unsigned int const denominator = 1 + blocks / maximumBlocks;
			blocks = ( blocks + ( denominator - 1 ) ) / denominator;
			BOOST_ASSERT( blocks <= maximumBlocks );
		}

		SparseKernelFindLargestScoreKernel<<< blocks, 256 >>>(
			reinterpret_cast< float*    >( deviceWork1 ),
			reinterpret_cast< boost::uint32_t* >( deviceWork2 ),
			deviceClusterHeaders,
			logMaximumClusterSize,
			clusters,
			classes,
			resultSize,
			logSortSize,
			regularization
		);

		std::pair< std::pair< float const*, boost::uint32_t const* >, unsigned int > deviceResult = FUFindLargest(
			deviceWork3,
			deviceWork4,
			reinterpret_cast< float*    >( deviceWork1 ),
			reinterpret_cast< boost::uint32_t* >( deviceWork2 ),
			workSize,
			resultSize,
			destinationSize,
			size
		);
		cudaMemcpy(
			destinationKeys,
			deviceResult.first.first,
			deviceResult.second * sizeof( float ),
			cudaMemcpyDeviceToHost
		);
		cudaMemcpy(
			destinationValues,
			deviceResult.first.second,
			deviceResult.second * sizeof( boost::uint32_t ),
			cudaMemcpyDeviceToHost
		);

		std::set< std::pair< float, boost::uint32_t > > maxima;
		for ( unsigned int ii = 0; ii < deviceResult.second; ++ii ) {

			if ( ( maxima.size() < resultSize ) || ( destinationKeys[ ii ] > maxima.begin()->first ) )
				maxima.insert( std::pair< float, boost::uint32_t >( destinationKeys[ ii ], destinationValues[ ii ] ) );
			while ( maxima.size() > resultSize )
				maxima.erase( maxima.begin() );
		}
		if ( maxima.size() != resultSize )
			throw std::runtime_error( "SparseKernelFindLargestScore: did not find the desired number of maxima" );
		{	unsigned int index = 0;
			std::set< std::pair< float, boost::uint32_t > >::const_iterator ii    = maxima.begin();
			std::set< std::pair< float, boost::uint32_t > >::const_iterator iiEnd = maxima.end();
			for ( ; ii != iiEnd; ++ii, ++index ) {

				destinationKeys[   index ] = ii->first;
				destinationValues[ index ] = ii->second;
			}
		}
	}
}




//============================================================================
//    SparseKernelFindLargestPositiveGradient function
//============================================================================


void SparseKernelFindLargestPositiveGradient(
	float* const destinationKeys,
	boost::uint32_t* const destinationValues,
	void* deviceWork1,
	void* deviceWork2,
	void* deviceWork3,
	void* deviceWork4,
	SparseKernelClusterHeader const* const deviceClusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const workSize,
	unsigned int const resultSize,
	unsigned int const destinationSize,
	float const regularization
)
{
	if ( ( resultSize < 1 ) || ( resultSize > 256 ) )
		throw std::runtime_error( "SparseKernelFindLargestPositiveGradient: result size must be between 1 and 256" );

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

	{	unsigned int const clusterBlocks = ( ( ( clusters << logMaximumClusterSize ) + 511 ) >> 9 );
		unsigned int const size = ( clusterBlocks << ( 9 - logSortSize ) ) * resultSize;
		if ( workSize < size * std::max( sizeof( float ), sizeof( boost::uint32_t ) ) )
			throw std::runtime_error( "SparseKernelFindLargestPositiveGradient: work buffer is too small!" );

		/*
			start out with the maximum possible number of blocks (one unit of work
			per thread), and divide by an integer (so that each thread is doing
			the same amount of work) to get below the target
		*/
		unsigned int blocks = clusterBlocks;
		{	unsigned int const maximumBlocks = 65535u;
			unsigned int const denominator = 1 + blocks / maximumBlocks;
			blocks = ( blocks + ( denominator - 1 ) ) / denominator;
			BOOST_ASSERT( blocks <= maximumBlocks );
		}

		SparseKernelFindLargestPositiveGradientKernel<<< blocks, 256 >>>(
			reinterpret_cast< float*    >( deviceWork1 ),
			reinterpret_cast< boost::uint32_t* >( deviceWork2 ),
			deviceClusterHeaders,
			logMaximumClusterSize,
			clusters,
			resultSize,
			logSortSize,
			regularization
		);

		std::pair< std::pair< float const*, boost::uint32_t const* >, unsigned int > deviceResult = FUFindLargest(
			deviceWork3,
			deviceWork4,
			reinterpret_cast< float*    >( deviceWork1 ),
			reinterpret_cast< boost::uint32_t* >( deviceWork2 ),
			workSize,
			resultSize,
			destinationSize,
			size
		);
		cudaMemcpy(
			destinationKeys,
			deviceResult.first.first,
			deviceResult.second * sizeof( float ),
			cudaMemcpyDeviceToHost
		);
		cudaMemcpy(
			destinationValues,
			deviceResult.first.second,
			deviceResult.second * sizeof( boost::uint32_t ),
			cudaMemcpyDeviceToHost
		);

		std::set< std::pair< float, boost::uint32_t > > maxima;
		for ( unsigned int ii = 0; ii < deviceResult.second; ++ii ) {

			if ( ( maxima.size() < resultSize ) || ( destinationKeys[ ii ] > maxima.begin()->first ) )
				maxima.insert( std::pair< float, boost::uint32_t >( destinationKeys[ ii ], destinationValues[ ii ] ) );
			while ( maxima.size() > resultSize )
				maxima.erase( maxima.begin() );
		}
		if ( maxima.size() != resultSize )
			throw std::runtime_error( "SparseKernelFindLargestPositiveGradient: did not find the desired number of maxima" );
		{	unsigned int index = 0;
			std::set< std::pair< float, boost::uint32_t > >::const_iterator ii    = maxima.begin();
			std::set< std::pair< float, boost::uint32_t > >::const_iterator iiEnd = maxima.end();
			for ( ; ii != iiEnd; ++ii, ++index ) {

				destinationKeys[   index ] = ii->first;
				destinationValues[ index ] = ii->second;
			}
		}
	}
}




//============================================================================
//    SparseKernelFindLargestNegativeGradient function
//============================================================================


void SparseKernelFindLargestNegativeGradient(
	float* const destinationKeys,
	boost::uint32_t* const destinationValues,
	void* deviceWork1,
	void* deviceWork2,
	void* deviceWork3,
	void* deviceWork4,
	SparseKernelClusterHeader const* const deviceClusterHeaders,
	unsigned int const logMaximumClusterSize,
	unsigned int const clusters,
	unsigned int const workSize,
	unsigned int const resultSize,
	unsigned int const destinationSize,
	float const regularization
)
{
	if ( ( resultSize < 1 ) || ( resultSize > 256 ) )
		throw std::runtime_error( "SparseKernelFindLargestNegativeGradient: result size must be between 1 and 256" );

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

	{	unsigned int const clusterBlocks = ( ( ( clusters << logMaximumClusterSize ) + 511 ) >> 9 );
		unsigned int const size = ( clusterBlocks << ( 9 - logSortSize ) ) * resultSize;
		if ( workSize < size * std::max( sizeof( float ), sizeof( boost::uint32_t ) ) )
			throw std::runtime_error( "SparseKernelFindLargestNegativeGradient: work buffer is too small!" );

		/*
			start out with the maximum possible number of blocks (one unit of work
			per thread), and divide by an integer (so that each thread is doing
			the same amount of work) to get below the target
		*/
		unsigned int blocks = clusterBlocks;
		{	unsigned int const maximumBlocks = 65535u;
			unsigned int const denominator = 1 + blocks / maximumBlocks;
			blocks = ( blocks + ( denominator - 1 ) ) / denominator;
			BOOST_ASSERT( blocks <= maximumBlocks );
		}

		SparseKernelFindLargestNegativeGradientKernel<<< blocks, 256 >>>(
			reinterpret_cast< float*    >( deviceWork1 ),
			reinterpret_cast< boost::uint32_t* >( deviceWork2 ),
			deviceClusterHeaders,
			logMaximumClusterSize,
			clusters,
			resultSize,
			logSortSize,
			regularization
		);

		std::pair< std::pair< float const*, boost::uint32_t const* >, unsigned int > deviceResult = FUFindLargest(
			deviceWork3,
			deviceWork4,
			reinterpret_cast< float*    >( deviceWork1 ),
			reinterpret_cast< boost::uint32_t* >( deviceWork2 ),
			workSize,
			resultSize,
			destinationSize,
			size
		);
		cudaMemcpy(
			destinationKeys,
			deviceResult.first.first,
			deviceResult.second * sizeof( float ),
			cudaMemcpyDeviceToHost
		);
		cudaMemcpy(
			destinationValues,
			deviceResult.first.second,
			deviceResult.second * sizeof( boost::uint32_t ),
			cudaMemcpyDeviceToHost
		);

		std::set< std::pair< float, boost::uint32_t > > maxima;
		for ( unsigned int ii = 0; ii < deviceResult.second; ++ii ) {

			if ( ( maxima.size() < resultSize ) || ( destinationKeys[ ii ] > maxima.begin()->first ) )
				maxima.insert( std::pair< float, boost::uint32_t >( destinationKeys[ ii ], destinationValues[ ii ] ) );
			while ( maxima.size() > resultSize )
				maxima.erase( maxima.begin() );
		}
		if ( maxima.size() != resultSize )
			throw std::runtime_error( "SparseKernelFindLargestNegativeGradient: did not find the desired number of maxima" );
		{	unsigned int index = 0;
			std::set< std::pair< float, boost::uint32_t > >::const_iterator ii    = maxima.begin();
			std::set< std::pair< float, boost::uint32_t > >::const_iterator iiEnd = maxima.end();
			for ( ; ii != iiEnd; ++ii, ++index ) {

				destinationKeys[   index ] = ii->first;
				destinationValues[ index ] = ii->second;
			}
		}
	}
}




}    // namespace CUDA




}    // namespace GTSVM
