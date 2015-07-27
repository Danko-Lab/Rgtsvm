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
	\file cuda_sparse_kernel.hpp
	\brief Front-end for CUDA kernel functions
*/




#ifndef __CUDA_SPARSE_KERNEL_HPP__
#define __CUDA_SPARSE_KERNEL_HPP__

#ifdef __cplusplus




#include "cuda_helpers.hpp"
#include "gtsvm.h"

#include <boost/cstdint.hpp>

#include <utility>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    SparseKernelClusterHeader structure
//============================================================================


struct SparseKernelClusterHeader {

	boost::uint32_t size;
	boost::uint32_t nonzeros;

	CUDA_FLOAT_DOUBLE* responses;
	boost::int32_t* labels;
	float* alphas;
	boost::uint32_t* nonzeroIndices;

	float* vectorsTranspose;
	float* vectorNormsSquared;
	float* vectorKernelNormsSquared;
};




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
	unsigned int const workSize,    // in bytes
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3
);




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
);




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
);




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
);




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
);




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
);




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
);




}    // namespace CUDA




}    // namespace GTSVM




#endif    /* __cplusplus */

#endif    /* __CUDA_SPARSE_KERNEL_HPP__ */
