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
	\file svm.hpp
	\brief definition of SVM class
*/




#ifndef __SVM_HPP__
#define __SVM_HPP__




#include "gtsvm.h"
#include "cuda.hpp"
#include "helpers.hpp"

#include <boost/shared_array.hpp>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/cstdint.hpp>

#include <map>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>




namespace GTSVM {




//============================================================================
//    SVM class
//============================================================================


struct SVM {

	BOOST_STATIC_ASSERT( sizeof( bool ) == 1 );
	BOOST_STATIC_ASSERT( sizeof( char ) == 1 );


	SVM();
	~SVM();


	void InitializeSparse(
		void const* const trainingVectors,    // order depends on columnMajor flag
		size_t const* const trainingVectorIndices,
		size_t const* const trainingVectorOffsets,
		GTSVM_Type trainingVectorsType,
		void const* const trainingLabels,
		GTSVM_Type trainingLabelsType,
		boost::uint32_t const rows,
		boost::uint32_t const columns,
		bool const columnMajor,
		bool const multiclass,
		float const regularization,
		GTSVM_Kernel const kernel,
		float const kernelParameter1,
		float const kernelParameter2,
		float const kernelParameter3,
		bool const biased,
		bool const smallClusters,
		unsigned int const activeClusters
	);

	void InitializeDense(
		void const* const trainingVectors,    // order depends on columnMajor flag
		GTSVM_Type trainingVectorsType,
		void const* const trainingLabels,
		GTSVM_Type trainingLabelsType,
		boost::uint32_t const rows,
		boost::uint32_t const columns,
		bool const columnMajor,
		bool const multiclass,
		float const regularization,
		GTSVM_Kernel const kernel,
		float const kernelParameter1,
		float const kernelParameter2,
		float const kernelParameter3,
		bool const biased,
		bool const smallClusters,
		unsigned int const activeClusters
	);

	void Load(
		char const* const filename,
		bool const smallClusters,
		unsigned int const activeClusters
	);

	void Save( char const* const filename ) const;

	void Shrink( bool const smallClusters, unsigned int const activeClusters );

	void DeinitializeDevice();
	void Deinitialize();


	inline unsigned int const GetRows()     const;
	inline unsigned int const GetColumns()  const;
	inline unsigned int const GetClasses()  const;
	inline unsigned int const GetNonzeros() const;

	inline float const GetRegularization() const;
	inline GTSVM_Kernel const GetKernel() const;
	inline float const GetKernelParameter1() const;
	inline float const GetKernelParameter2() const;
	inline float const GetKernelParameter3() const;
	inline bool const GetBiased() const;

	inline CUDA_FLOAT_DOUBLE const GetBias() const;


	void GetTrainingVectorsSparse(
		void* const trainingVectors,    // order depends on the columnMajor flag
		size_t* const trainingVectorIndices,
		size_t* const trainingVectorOffsets,
		GTSVM_Type trainingVectorsType,
		bool const columnMajor
	) const;

	void GetTrainingVectorsDense(
		void* const trainingVectors,    // order depends on the columnMajor flag
		GTSVM_Type trainingVectorsType,
		bool const columnMajor
	) const;

	void GetTrainingLabels(
		void* const trainingLabels,
		GTSVM_Type trainingLabelsType
	) const;

	void GetTrainingResponses(
		void* const trainingResponses,
		GTSVM_Type trainingResponsesType,
		bool const columnMajor
	) const;

	void GetAlphas(
		void* const trainingAlphas,
		GTSVM_Type trainingAlphasType,
		bool const columnMajor
	) const;


	void SetAlphas(
		void const* const trainingAlphas,
		GTSVM_Type trainingAlphasType,
		bool const columnMajor
	);


	void Recalculate();

	void Restart(
		float const regularization,
		GTSVM_Kernel const kernel,
		float const kernelParameter1,
		float const kernelParameter2,
		float const kernelParameter3,
		bool const biased
	);

	std::pair< CUDA_FLOAT_DOUBLE, CUDA_FLOAT_DOUBLE > const Optimize( unsigned int const iterations );


	void ClassifySparse(
		void* const result,
		GTSVM_Type resultType,
		void const* const vectors,    // order depends on columnMajor flag
		size_t const* const vectorIndices,
		size_t const* const vectorOffsets,
		GTSVM_Type vectorsType,
		unsigned int const rows,
		unsigned int const columns,
		bool const columnMajor
	);

	void ClassifyDense(
		void* const result,
		GTSVM_Type resultType,
		void const* const vectors,    // order depends on columnMajor flag
		GTSVM_Type vectorsType,
		unsigned int const rows,
		unsigned int const columns,
		bool const columnMajor
	);


private:

	void Cleanup();

	void ClusterTrainingVectors(
		bool const smallClusters,
		unsigned int activeClusters
	);

	void InitializeDevice();

	void UpdateResponses();


	bool const IterateUnbiasedBinary();
	bool const IterateBiasedBinary();
	bool const IterateUnbiasedMulticlass();


	typedef std::vector< std::pair< unsigned int, float > > SparseVector;


	bool m_constructed;
	bool m_initializedHost;
	bool m_initializedDevice;
	bool m_updatedResponses;

	boost::uint32_t m_rows;
	boost::uint32_t m_columns;
	boost::uint32_t m_classes;

	boost::shared_array< SparseVector > m_trainingVectors;
	boost::shared_array< boost::int32_t > m_trainingLabels;
	boost::shared_array< float > m_trainingVectorNormsSquared;
	boost::shared_array< float > m_trainingVectorKernelNormsSquared;

	float m_regularization;
	GTSVM_Kernel m_kernel;
	float m_kernelParameter1;
	float m_kernelParameter2;
	float m_kernelParameter3;
	bool m_biased;

	CUDA_FLOAT_DOUBLE m_bias;
	boost::shared_array< double > m_trainingResponses;
	boost::shared_array< float > m_trainingAlphas;

	unsigned int m_logMaximumClusterSize;
	unsigned int m_clusters;
	std::vector< std::vector< unsigned int > > m_clusterIndices;
	std::vector< std::vector< unsigned int > > m_clusterNonzeroIndices;

	size_t m_foundSize;
	boost::uint32_t m_foundIndices[ 32 ];
	float* m_foundKeys;
	boost::uint32_t* m_foundValues;

	float* m_batchVectorsTranspose;
	float* m_batchVectorNormsSquared;
	CUDA_FLOAT_DOUBLE* m_batchResponses;
	float* m_batchAlphas;
	boost::uint32_t* m_batchIndices;

	boost::shared_array< double > m_batchSubmatrix;

	float* m_deviceBatchVectorsTranspose;
	float* m_deviceBatchVectorNormsSquared;
	CUDA_FLOAT_DOUBLE* m_deviceBatchResponses;
	float* m_deviceBatchAlphas;
	boost::uint32_t* m_deviceBatchIndices;
	boost::int32_t* m_deviceTrainingLabels;
	float* m_deviceTrainingVectorNormsSquared;
	float* m_deviceTrainingVectorKernelNormsSquared;
	CUDA_FLOAT_DOUBLE* m_deviceTrainingResponses;
	float* m_deviceTrainingAlphas;

	boost::uint32_t* m_deviceNonzeroIndices;
	float* m_deviceTrainingVectorsTranspose;
	CUDA::SparseKernelClusterHeader* m_deviceClusterHeaders;
	boost::uint32_t* m_deviceClusterSizeSums;

	size_t m_workSize;
	void* m_deviceWork[ 4 ];
};




//============================================================================
//    SVM inline methods
//============================================================================


unsigned int const SVM::GetRows() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_rows;
}


unsigned int const SVM::GetColumns() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_columns;
}


unsigned int const SVM::GetClasses() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_classes;
}


unsigned int const SVM::GetNonzeros() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	unsigned int nonzeros = 0;
	for ( unsigned int ii = 0; ii < m_rows; ++ii )
		nonzeros += m_trainingVectors[ ii ].size();

	return nonzeros;
}


float const SVM::GetRegularization() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_regularization;
}


GTSVM_Kernel const SVM::GetKernel() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_kernel;
}


float const SVM::GetKernelParameter1() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_kernelParameter1;
}


float const SVM::GetKernelParameter2() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_kernelParameter2;
}


float const SVM::GetKernelParameter3() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_kernelParameter3;
}


bool const SVM::GetBiased() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_biased;
}


CUDA_FLOAT_DOUBLE const SVM::GetBias() const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	return m_bias;
}




}    // namespace GTSVM




#endif    /* __SVM_HPP__ */
