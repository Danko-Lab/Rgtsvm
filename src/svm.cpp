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
	\file svm.cpp
	\brief implementation of SVM class
*/




#include "headers.hpp"




namespace GTSVM {




namespace {




//============================================================================
//    Kernel functors
//============================================================================


template< int t_Kernel >
struct Kernel { };


template<>
struct Kernel< GTSVM_KERNEL_GAUSSIAN > {

	static inline double Calculate(
		float innerProduct,
		float normSquared1,
		float normSquared2,
		float kernelParameter1,
		float kernelParameter2,
		float kernelParameter3
	)
	{
		BOOST_ASSERT( kernelParameter1 > 0 );
		return std::exp( kernelParameter1 * ( 2 * innerProduct - normSquared1 - normSquared2 ) );
	}
};


template<>
struct Kernel< GTSVM_KERNEL_POLYNOMIAL > {

	static inline double Calculate(
		float innerProduct,
		float normSquared1,
		float normSquared2,
		float kernelParameter1,
		float kernelParameter2,
		float kernelParameter3
	)
	{
		BOOST_ASSERT( kernelParameter3 > 0 );
		return std::pow( kernelParameter1 * innerProduct + kernelParameter2, kernelParameter3 );
	}
};


template<>
struct Kernel< GTSVM_KERNEL_SIGMOID > {

	static inline double Calculate(
		float innerProduct,
		float normSquared1,
		float normSquared2,
		float kernelParameter1,
		float kernelParameter2,
		float kernelParameter3
	)
	{
		double const exponent = std::exp( 2 * ( kernelParameter1 * innerProduct + kernelParameter2 ) );
		return( ( exponent - 1 ) / ( exponent + 1 ) );
	}
};




//============================================================================
//    SVM_ConvertHelper helper class
//============================================================================


template< typename t_DestinationType >
struct SVM_ConvertHelper {

	template< typename t_SourceType >
	static inline t_DestinationType const Convert( t_SourceType const source ) {

		return static_cast< t_DestinationType >( source );
	}
};


template<>
struct SVM_ConvertHelper< bool > {

	template< typename t_SourceType >
	static inline bool const Convert( t_SourceType const source ) {

		return( source > 0 );
	}

	static inline bool const Convert( bool const source ) {

		return source;
	}
};




//============================================================================
//    SVM_Memcpy helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_Memcpy_Helper( t_DestinationType* const destination, t_SourceType const* const source, size_t const count ) {

	t_SourceType const* ii    = source;
	t_SourceType const* iiEnd = source + count;

	t_DestinationType* jj = destination;

	for ( ; ii != iiEnd; ++ii, ++jj )
		*jj = SVM_ConvertHelper< t_DestinationType >::Convert( *ii );
}


template< typename t_Type >
static inline void SVM_Memcpy( t_Type* const destination, void const* const source, size_t const sourceOffset, GTSVM_Type const type, size_t const count ) {

	switch( type ) {
		case GTSVM_TYPE_BOOL:   SVM_Memcpy_Helper( destination, reinterpret_cast< bool const*            >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_FLOAT:  SVM_Memcpy_Helper( destination, reinterpret_cast< float const*           >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_DOUBLE: SVM_Memcpy_Helper( destination, reinterpret_cast< double const*          >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_INT8:   SVM_Memcpy_Helper( destination, reinterpret_cast< boost::int8_t const*   >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_INT16:  SVM_Memcpy_Helper( destination, reinterpret_cast< boost::int16_t const*  >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_INT32:  SVM_Memcpy_Helper( destination, reinterpret_cast< boost::int32_t const*  >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_INT64:  SVM_Memcpy_Helper( destination, reinterpret_cast< boost::int64_t const*  >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_UINT8:  SVM_Memcpy_Helper( destination, reinterpret_cast< boost::uint8_t const*  >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_UINT16: SVM_Memcpy_Helper( destination, reinterpret_cast< boost::uint16_t const* >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_UINT32: SVM_Memcpy_Helper( destination, reinterpret_cast< boost::uint32_t const* >( source ) + sourceOffset, count ); break;
		case GTSVM_TYPE_UINT64: SVM_Memcpy_Helper( destination, reinterpret_cast< boost::uint64_t const* >( source ) + sourceOffset, count ); break;
		default: throw std::runtime_error( "Unknown type" );
	}
}




//============================================================================
//    SVM_Memcpy2d helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_Memcpy2d_Helper( t_DestinationType* const destination, t_SourceType const* const source, size_t const rows, size_t const columns ) {

	unsigned int index1 = 0;
	for ( unsigned int ii = 0; ii < rows; ++ii ) {

		unsigned int index2 = ii;
		for ( unsigned int jj = 0; jj < columns; ++jj ) {

			destination[ index1 ] = SVM_ConvertHelper< t_DestinationType >::Convert( source[ index2 ] );

			++index1;
			index2 += rows;
		}
	}
}


template< typename t_Type >
static inline void SVM_Memcpy2d( t_Type* const destination, void const* const source, GTSVM_Type const type, size_t const rows, size_t const columns, bool const transpose = false ) {

	if ( transpose ) {

		switch( type ) {
			case GTSVM_TYPE_BOOL:   SVM_Memcpy2d_Helper( destination, reinterpret_cast< bool const*            >( source ), rows, columns ); break;
			case GTSVM_TYPE_FLOAT:  SVM_Memcpy2d_Helper( destination, reinterpret_cast< float const*           >( source ), rows, columns ); break;
			case GTSVM_TYPE_DOUBLE: SVM_Memcpy2d_Helper( destination, reinterpret_cast< double const*          >( source ), rows, columns ); break;
			case GTSVM_TYPE_INT8:   SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::int8_t const*   >( source ), rows, columns ); break;
			case GTSVM_TYPE_INT16:  SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::int16_t const*  >( source ), rows, columns ); break;
			case GTSVM_TYPE_INT32:  SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::int32_t const*  >( source ), rows, columns ); break;
			case GTSVM_TYPE_INT64:  SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::int64_t const*  >( source ), rows, columns ); break;
			case GTSVM_TYPE_UINT8:  SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::uint8_t const*  >( source ), rows, columns ); break;
			case GTSVM_TYPE_UINT16: SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::uint16_t const* >( source ), rows, columns ); break;
			case GTSVM_TYPE_UINT32: SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::uint32_t const* >( source ), rows, columns ); break;
			case GTSVM_TYPE_UINT64: SVM_Memcpy2d_Helper( destination, reinterpret_cast< boost::uint64_t const* >( source ), rows, columns ); break;
			default: throw std::runtime_error( "Unknown type" );
		}
	}
	else
		SVM_Memcpy( destination, source, 0, type, rows * columns );
}




//============================================================================
//    SVM_SparseMemcpy2d helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_SparseMemcpy2d_Helper( std::vector< std::pair< unsigned int, t_DestinationType > >* const destination, t_SourceType const* const source, size_t const rows, size_t const columns, bool const transpose ) {

	if ( transpose ) {

		for ( unsigned int ii = 0; ii < rows; ++ii ) {

			unsigned int index = ii;
			for ( unsigned int jj = 0; jj < columns; ++jj ) {

				t_DestinationType const value = SVM_ConvertHelper< t_DestinationType >::Convert( source[ index ] );
				if ( value != 0 )
					destination[ ii ].push_back( std::pair< unsigned int, t_DestinationType >( jj, value ) );
				index += rows;
			}
		}
	}
	else {

		unsigned int index = 0;
		for ( unsigned int ii = 0; ii < rows; ++ii ) {

			for ( unsigned int jj = 0; jj < columns; ++jj ) {

				t_DestinationType const value = SVM_ConvertHelper< t_DestinationType >::Convert( source[ index ] );
				if ( value != 0 )
					destination[ ii ].push_back( std::pair< unsigned int, t_DestinationType >( jj, value ) );
				++index;
			}
		}
	}
}


template< typename t_Type >
static inline void SVM_SparseMemcpy2d( std::vector< std::pair< unsigned int, t_Type > >* const destination, void const* const source, GTSVM_Type const type, size_t const rows, size_t const columns, bool const transpose = false ) {

	switch( type ) {
		case GTSVM_TYPE_BOOL:   SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< bool const*            >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_FLOAT:  SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< float const*           >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_DOUBLE: SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< double const*          >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_INT8:   SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int8_t const*   >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_INT16:  SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int16_t const*  >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_INT32:  SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int32_t const*  >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_INT64:  SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int64_t const*  >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT8:  SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint8_t const*  >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT16: SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint16_t const* >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT32: SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint32_t const* >( source ), rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT64: SVM_SparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint64_t const* >( source ), rows, columns, transpose ); break;
		default: throw std::runtime_error( "Unknown type" );
	}
}




//============================================================================
//    SVM_SparseSparseMemcpy2d helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_SparseSparseMemcpy2d_Helper( std::vector< std::pair< unsigned int, t_DestinationType > >* const destination, t_SourceType const* const source, size_t const* const sourceIndices, size_t const* const sourceOffsets, size_t const rows, size_t const columns, bool const transpose ) {

	if ( transpose ) {

		boost::shared_array< unsigned int > indices( new unsigned int[ columns ] );
		for ( unsigned int ii = 0; ii < columns; ++ii )
			indices[ ii ] = sourceOffsets[ ii ];

		std::priority_queue< std::pair< int, int > > queue;
		for ( unsigned int ii = 0; ii < columns; ++ii )
			if ( indices[ ii ] < sourceOffsets[ ii + 1 ] )
				queue.push( std::pair< int, int >( -static_cast< int >( sourceIndices[ indices[ ii ] ] ), -static_cast< int >( ii ) ) );

		for ( unsigned int ii = 0; ii < rows; ++ii ) {

			while ( ! queue.empty() ) {

				std::pair< int, int > const head = queue.top();
				unsigned int const row    = -head.first;
				unsigned int const column = -head.second;
				BOOST_ASSERT( ( row >= ii ) && ( row < rows ) );
				BOOST_ASSERT( column < columns );
				if ( row != ii )
					break;

				t_DestinationType const value = SVM_ConvertHelper< t_DestinationType >::Convert( source[ indices[ column ] ] );
				if ( value != 0 )
					destination[ ii ].push_back( std::pair< unsigned int, t_DestinationType >( column, value ) );

				queue.pop();
				++indices[ column ];
				if ( indices[ column ] < sourceOffsets[ column + 1 ] )
					queue.push( std::pair< int, int >( -static_cast< int >( sourceIndices[ indices[ column ] ] ), -static_cast< int >( column ) ) );
			}
		}
	}
	else {

		for ( unsigned int ii = 0; ii < rows; ++ii ) {

			for ( unsigned int jj = sourceOffsets[ ii ]; jj < sourceOffsets[ ii + 1 ]; ++jj ) {

				t_DestinationType const value = SVM_ConvertHelper< t_DestinationType >::Convert( source[ jj ] );
				if ( value != 0 )
					destination[ ii ].push_back( std::pair< unsigned int, t_DestinationType >( sourceIndices[ jj ], value ) );
			}
		}
	}
}


template< typename t_Type >
static inline void SVM_SparseSparseMemcpy2d( std::vector< std::pair< unsigned int, t_Type > >* const destination, void const* const source, size_t const* const sourceIndices, size_t const* const sourceOffsets, GTSVM_Type const type, size_t const rows, size_t const columns, bool const transpose = false ) {

	switch( type ) {
		case GTSVM_TYPE_BOOL:   SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< bool const*            >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_FLOAT:  SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< float const*           >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_DOUBLE: SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< double const*          >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT8:   SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int8_t const*   >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT16:  SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int16_t const*  >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT32:  SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int32_t const*  >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT64:  SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::int64_t const*  >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT8:  SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint8_t const*  >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT16: SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint16_t const* >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT32: SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint32_t const* >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT64: SVM_SparseSparseMemcpy2d_Helper( destination, reinterpret_cast< boost::uint64_t const* >( source ), sourceIndices, sourceOffsets, rows, columns, transpose ); break;
		default: throw std::runtime_error( "Unknown type" );
	}
}




//============================================================================
//    SVM_ReverseMemcpy helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_ReverseMemcpy_Helper( t_DestinationType* const destination, t_SourceType const* const source, size_t const count ) {

	t_SourceType const* ii    = source;
	t_SourceType const* iiEnd = source + count;

	t_DestinationType* jj = destination;

	for ( ; ii != iiEnd; ++ii, ++jj )
		*jj = SVM_ConvertHelper< t_DestinationType >::Convert( *ii );

}


template< typename t_Type >
static inline void SVM_ReverseMemcpy( void* const destination, size_t const destinationOffset, GTSVM_Type const type, t_Type const* const source, size_t const count ) {

	switch( type ) {
		case GTSVM_TYPE_BOOL:   SVM_ReverseMemcpy_Helper( reinterpret_cast< bool*            >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_FLOAT:  SVM_ReverseMemcpy_Helper( reinterpret_cast< float*           >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_DOUBLE: SVM_ReverseMemcpy_Helper( reinterpret_cast< double*          >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_INT8:   SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::int8_t*   >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_INT16:  SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::int16_t*  >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_INT32:  SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::int32_t*  >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_INT64:  SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::int64_t*  >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_UINT8:  SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::uint8_t*  >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_UINT16: SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::uint16_t* >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_UINT32: SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::uint32_t* >( destination ) + destinationOffset, source, count ); break;
		case GTSVM_TYPE_UINT64: SVM_ReverseMemcpy_Helper( reinterpret_cast< boost::uint64_t* >( destination ) + destinationOffset, source, count ); break;
		default: throw std::runtime_error( "Unknown type" );
	}
}




//============================================================================
//    SVM_ReverseMemcpy2d helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_ReverseMemcpy2d_Helper( t_DestinationType* const destination, t_SourceType const* const source, size_t const rows, size_t const columns ) {

	unsigned int index1 = 0;
	for ( unsigned int ii = 0; ii < rows; ++ii ) {

		unsigned int index2 = ii;
		for ( unsigned int jj = 0; jj < columns; ++jj ) {

			destination[ index2 ] = SVM_ConvertHelper< t_DestinationType >::Convert( source[ index1 ] );

			++index1;
			index2 += rows;
		}
	}
}


template< typename t_Type >
static inline void SVM_ReverseMemcpy2d( void* const destination, GTSVM_Type const type, t_Type const* const source, size_t const rows, size_t const columns, bool const transpose = false ) {

	if ( transpose ) {

		switch( type ) {
			case GTSVM_TYPE_BOOL:   SVM_ReverseMemcpy2d_Helper( reinterpret_cast< bool*            >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_FLOAT:  SVM_ReverseMemcpy2d_Helper( reinterpret_cast< float*           >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_DOUBLE: SVM_ReverseMemcpy2d_Helper( reinterpret_cast< double*          >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_INT8:   SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::int8_t*   >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_INT16:  SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::int16_t*  >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_INT32:  SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::int32_t*  >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_INT64:  SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::int64_t*  >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_UINT8:  SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::uint8_t*  >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_UINT16: SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::uint16_t* >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_UINT32: SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::uint32_t* >( destination ), source, rows, columns ); break;
			case GTSVM_TYPE_UINT64: SVM_ReverseMemcpy2d_Helper( reinterpret_cast< boost::uint64_t* >( destination ), source, rows, columns ); break;
			default: throw std::runtime_error( "Unknown type" );
		}
	}
	else
		SVM_ReverseMemcpy( destination, 0, type, source, rows * columns );
}




//============================================================================
//    SVM_SparseReverseMemcpy2d helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_SparseReverseMemcpy2d_Helper( t_DestinationType* const destination, std::vector< std::pair< unsigned int, t_SourceType > > const* const source, size_t const rows, size_t const columns, bool const transpose ) {

	if ( transpose ) {

		for ( unsigned int ii = 0; ii < rows; ++ii ) {

			unsigned int index = ii;

			unsigned int kk = 0;
			typename std::vector< std::pair< unsigned int, t_SourceType > >::const_iterator jj    = source[ ii ].begin();
			typename std::vector< std::pair< unsigned int, t_SourceType > >::const_iterator jjEnd = source[ ii ].end();
			for ( ; jj != jjEnd; ++jj ) {

				BOOST_ASSERT( jj->first < columns );
				for ( ; kk < jj->first; ++kk ) {

					destination[ index ] = 0;
					index += rows;
				}
				destination[ index ] = SVM_ConvertHelper< t_DestinationType >::Convert( jj->second );
				index += rows;
				++kk;
			}
			for ( ; kk < columns; ++kk ) {

				destination[ index ] = 0;
				index += rows;
			}
		}
	}
	else {

		unsigned int index = 0;
		for ( unsigned int ii = 0; ii < rows; ++ii ) {

			unsigned int kk = 0;
			typename std::vector< std::pair< unsigned int, t_SourceType > >::const_iterator jj    = source[ ii ].begin();
			typename std::vector< std::pair< unsigned int, t_SourceType > >::const_iterator jjEnd = source[ ii ].end();
			for ( ; jj != jjEnd; ++jj ) {

				BOOST_ASSERT( jj->first < columns );
				for ( ; kk < jj->first; ++kk ) {

					destination[ index ] = 0;
					++index;
				}
				destination[ index ] = SVM_ConvertHelper< t_DestinationType >::Convert( jj->second );
				++index;
				++kk;
			}
			for ( ; kk < columns; ++kk ) {

				destination[ index ] = 0;
				++index;
			}
		}
	}
}


template< typename t_Type >
static inline void SVM_SparseReverseMemcpy2d( void* const destination, GTSVM_Type const type, std::vector< std::pair< unsigned int, t_Type > > const* const source, size_t const rows, size_t const columns, bool const transpose = false ) {

	switch( type ) {
		case GTSVM_TYPE_BOOL:   SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< bool*            >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_FLOAT:  SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< float*           >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_DOUBLE: SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< double*          >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT8:   SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int8_t*   >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT16:  SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int16_t*  >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT32:  SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int32_t*  >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT64:  SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int64_t*  >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT8:  SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint8_t*  >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT16: SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint16_t* >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT32: SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint32_t* >( destination ), source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT64: SVM_SparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint64_t* >( destination ), source, rows, columns, transpose ); break;
		default: throw std::runtime_error( "Unknown type" );
	}
}




//============================================================================
//    SVM_SparseSparseReverseMemcpy2d helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_SparseSparseReverseMemcpy2d_Helper( t_DestinationType* const destination, size_t* const destinationIndices, size_t* const destinationOffsets, std::vector< std::pair< unsigned int, t_SourceType > > const* const source, size_t const rows, size_t const columns, bool const transpose ) {

	if ( transpose ) {

		boost::shared_array< unsigned int > indices( new unsigned int[ rows ] );
		std::fill( indices.get(), indices.get() + rows, 0 );

		std::priority_queue< std::pair< int, int > > queue;
		for ( unsigned int ii = 0; ii < rows; ++ii )
			if ( indices[ ii ] < source[ ii ].size() )
				queue.push( std::pair< int, int >( -static_cast< int >( source[ ii ][ indices[ ii ] ].first ), -static_cast< int >( ii ) ) );

		unsigned int index = 0;
		for ( unsigned int ii = 0; ii < columns; ++ii ) {

			destinationOffsets[ ii ] = index;

			while ( ! queue.empty() ) {

				std::pair< int, int > const head = queue.top();
				unsigned int const column = -head.first;
				unsigned int const row    = -head.second;
				BOOST_ASSERT( ( column >= ii ) && ( column < columns ) );
				BOOST_ASSERT( row < rows );
				if ( column != ii )
					break;

				destination[        index ] = SVM_ConvertHelper< t_DestinationType >::Convert( source[ row ][ indices[ row ] ].second );
				destinationIndices[ index ] = row;
				++index;

				queue.pop();
				++indices[ row ];
				if ( indices[ row ] < source[ row ].size() )
					queue.push( std::pair< int, int >( -static_cast< int >( source[ row ][ indices[ row ] ].first ), -static_cast< int >( row ) ) );
			}
		}
		destinationOffsets[ columns ] = index;
	}
	else {

		unsigned int index = 0;
		for ( unsigned int ii = 0; ii < rows; ++ii ) {

			destinationOffsets[ ii ] = index;

			typename std::vector< std::pair< unsigned int, t_SourceType > >::const_iterator jj    = source[ ii ].begin();
			typename std::vector< std::pair< unsigned int, t_SourceType > >::const_iterator jjEnd = source[ ii ].end();
			for ( ; jj != jjEnd; ++jj ) {

				BOOST_ASSERT( jj->first < columns );
				destination[        index ] = SVM_ConvertHelper< t_DestinationType >::Convert( jj->second );
				destinationIndices[ index ] = jj->first;
				++index;
			}
		}
		destinationOffsets[ rows ] = index;
	}
}


template< typename t_Type >
static inline void SVM_SparseSparseReverseMemcpy2d( void* const destination, size_t* const destinationIndices, size_t* const destinationOffsets, GTSVM_Type const type, std::vector< std::pair< unsigned int, t_Type > > const* const source, size_t const rows, size_t const columns, bool const transpose = false ) {

	switch( type ) {
		case GTSVM_TYPE_BOOL:   SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< bool*            >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_FLOAT:  SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< float*           >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_DOUBLE: SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< double*          >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT8:   SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int8_t*   >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT16:  SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int16_t*  >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT32:  SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int32_t*  >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_INT64:  SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::int64_t*  >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT8:  SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint8_t*  >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT16: SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint16_t* >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT32: SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint32_t* >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		case GTSVM_TYPE_UINT64: SVM_SparseSparseReverseMemcpy2d_Helper( reinterpret_cast< boost::uint64_t* >( destination ), destinationIndices, destinationOffsets, source, rows, columns, transpose ); break;
		default: throw std::runtime_error( "Unknown type" );
	}
}




//============================================================================
//    SVM_MemcpyStride helper function
//============================================================================


template< typename t_DestinationType, typename t_SourceType >
static inline void SVM_MemcpyStride_Helper( t_DestinationType* const destination, size_t const destinationStride, t_SourceType const* const source, size_t const sourceStride, size_t const count ) {

	t_DestinationType* pDestination = destination;
	t_SourceType const* pSource = source;
	for ( unsigned int ii = 0; ii < count; ++ii ) {

		*pDestination = SVM_ConvertHelper< t_DestinationType >::Convert( *pSource );
		pDestination += destinationStride;
		pSource += sourceStride;
	}
}


template< typename t_Type >
static inline void SVM_MemcpyStride( t_Type* const destination, size_t const destinationStride, void const* const source, size_t const sourceOffset, size_t const sourceStride, GTSVM_Type const type, size_t const count ) {

	switch( type ) {
		case GTSVM_TYPE_BOOL:   SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< bool const*            >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_FLOAT:  SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< float const*           >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_DOUBLE: SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< double const*          >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_INT8:   SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::int8_t const*   >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_INT16:  SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::int16_t const*  >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_INT32:  SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::int32_t const*  >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_INT64:  SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::int64_t const*  >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_UINT8:  SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::uint8_t const*  >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_UINT16: SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::uint16_t const* >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_UINT32: SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::uint32_t const* >( source ) + sourceOffset, sourceStride, count ); break;
		case GTSVM_TYPE_UINT64: SVM_MemcpyStride_Helper( destination, destinationStride, reinterpret_cast< boost::uint64_t const* >( source ) + sourceOffset, sourceStride, count ); break;
		default: throw std::runtime_error( "Unknown type" );
	}
}




}    // anonymous namespace




//============================================================================
//    SVM methods
//============================================================================


SVM::SVM() :
	m_constructed( true ),
	m_initializedHost( false ),
	m_initializedDevice( false ),
	m_updatedResponses( true ),
	m_foundKeys( NULL ),
	m_foundValues( NULL ),
	m_batchVectorsTranspose( NULL ),
	m_batchVectorNormsSquared( NULL ),
	m_batchResponses( NULL ),
	m_batchAlphas( NULL ),
	m_batchIndices( NULL ),
	m_deviceBatchVectorsTranspose( NULL ),
	m_deviceBatchVectorNormsSquared( NULL ),
	m_deviceBatchResponses( NULL ),
	m_deviceBatchAlphas( NULL ),
	m_deviceBatchIndices( NULL ),
	m_deviceTrainingLabels( NULL ),
	m_deviceTrainingVectorNormsSquared( NULL ),
	m_deviceTrainingVectorKernelNormsSquared( NULL ),
	m_deviceTrainingResponses( NULL ),
	m_deviceTrainingAlphas( NULL ),
	m_deviceNonzeroIndices( NULL ),
	m_deviceTrainingVectorsTranspose( NULL ),
	m_deviceClusterHeaders( NULL ),
	m_deviceClusterSizeSums( NULL )
{
	for ( unsigned int ii = 0; ii < ARRAYLENGTH( m_deviceWork ); ++ii )
		m_deviceWork[ ii ] = NULL;

	try {

		m_foundSize = 4096;
		CUDA_VERIFY(
			"Failed to allocate space for found keys on host",
			cudaMallocHost( &m_foundKeys, m_foundSize * sizeof( float ) )
		);
		CUDA_VERIFY(
			"Failed to allocate space for found values on host",
			cudaMallocHost( &m_foundValues, m_foundSize * sizeof( boost::uint32_t ) )
		);

		CUDA_VERIFY(
			"Failed to allocate space for batch squared norms on host",
			cudaMallocHost( &m_batchVectorNormsSquared, 16 * sizeof( float ) )
		);
		CUDA_VERIFY(
			"Failed to allocate space for batch squared norms on device",
			cudaMalloc( reinterpret_cast< void** >( &m_deviceBatchVectorNormsSquared ), 16 * sizeof( float ) )
		);

		m_batchSubmatrix = boost::shared_array< double >( new double[ 256 ] );
	}
	catch( ... ) {

		Cleanup();    // try to keep this structure in a valid state, if possible
		throw;
	}
}


SVM::~SVM() {

	Cleanup();
}


void SVM::InitializeSparse(
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
)
{
	if ( ! m_constructed )
		throw std::runtime_error( "SVM has not been successfully constructed" );
	if ( m_initializedHost )
		Deinitialize();
	BOOST_ASSERT( ! m_initializedDevice );
	BOOST_ASSERT( ! m_initializedHost );
	m_initializedHost = true;

	try {

		m_rows = rows;
		m_columns = columns;

		m_trainingVectors = boost::shared_array< SparseVector >( new SparseVector[ m_rows ] );
		m_trainingLabels = boost::shared_array< boost::int32_t >( new boost::int32_t[ m_rows ] );
		SVM_SparseSparseMemcpy2d( m_trainingVectors.get(), trainingVectors, trainingVectorIndices, trainingVectorOffsets, trainingVectorsType, m_rows, m_columns, columnMajor );
		SVM_Memcpy( m_trainingLabels.get(), trainingLabels, 0, trainingLabelsType, m_rows );

		if ( multiclass ) {

			int maximumLabel = 0;
			for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

				int const label = m_trainingLabels[ ii ];
				if ( label < 0 )
					throw std::runtime_error( "multiclass labels must be nonnegative" );
				if ( label > maximumLabel )
					maximumLabel = label;
			}
			if ( maximumLabel > 65535 )
				throw std::runtime_error( "multiclass labels cannot exceed 65535" );

			m_classes = maximumLabel + 1;

			boost::shared_array< bool > present( new bool[ m_classes ] );
			std::fill( present.get(), present.get() + m_classes, false );
			for ( unsigned int ii = 0; ii < m_rows; ++ii )
				present[ m_trainingLabels[ ii ] ] = true;
			for ( unsigned int ii = 0; ii < m_classes; ++ii )
				if ( ! present[ ii ] )
					throw std::runtime_error( "at least one example of each label in {0,1,...,max} must be present in training set" );
		}
		else {

			m_classes = 1;

			bool present[ 2 ] = { false, false };
			for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

				if ( m_trainingLabels[ ii ] > 0 ) {

					m_trainingLabels[ ii ] = 1;
					present[ 1 ] = true;
				}
				else {

					m_trainingLabels[ ii ] = -1;
					present[ 0 ] = true;
				}
			}
			if ( ( ! present[ 0 ] ) || ( ! present[ 1 ] ) )
				throw std::runtime_error( "at least one positive and negative example must be present in training set" );
		}

		m_trainingVectorNormsSquared       = boost::shared_array< float >( new float[ m_rows ] );
		m_trainingVectorKernelNormsSquared = boost::shared_array< float >( new float[ m_rows ] );

		m_trainingResponses = boost::shared_array< double >( new double[ m_rows * m_classes ] );
		m_trainingAlphas = boost::shared_array< float >( new float[ m_rows * m_classes ] );

		ClusterTrainingVectors( smallClusters, activeClusters );

		Restart( regularization, kernel, kernelParameter1, kernelParameter2, kernelParameter3, biased );
	}
	catch( ... ) {

		Deinitialize();    // try to keep this structure in a valid state, if possible
		throw;
	}
}


void SVM::InitializeDense(
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
)
{
	if ( ! m_constructed )
		throw std::runtime_error( "SVM has not been successfully constructed" );
	if ( m_initializedHost )
		Deinitialize();
	BOOST_ASSERT( ! m_initializedDevice );
	BOOST_ASSERT( ! m_initializedHost );
	m_initializedHost = true;

	try {

		m_rows = rows;
		m_columns = columns;

		m_trainingVectors = boost::shared_array< SparseVector >( new SparseVector[ m_rows ] );
		m_trainingLabels = boost::shared_array< boost::int32_t >( new boost::int32_t[ m_rows ] );
		SVM_SparseMemcpy2d( m_trainingVectors.get(), trainingVectors, trainingVectorsType, m_rows, m_columns, columnMajor );
		SVM_Memcpy( m_trainingLabels.get(), trainingLabels, 0, trainingLabelsType, m_rows );

		if ( multiclass ) {

			int maximumLabel = 0;
			for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

				int const label = m_trainingLabels[ ii ];
				if ( label < 0 )
					throw std::runtime_error( "multiclass labels must be nonnegative" );
				if ( label > maximumLabel )
					maximumLabel = label;
			}
			if ( maximumLabel > 65535 )
				throw std::runtime_error( "multiclass labels cannot exceed 65535" );

			m_classes = maximumLabel + 1;

			boost::shared_array< bool > present( new bool[ m_classes ] );
			std::fill( present.get(), present.get() + m_classes, false );
			for ( unsigned int ii = 0; ii < m_rows; ++ii )
				present[ m_trainingLabels[ ii ] ] = true;
			for ( unsigned int ii = 0; ii < m_classes; ++ii )
				if ( ! present[ ii ] )
					throw std::runtime_error( "at least one example of each label in {0,1,...,max} must be present in training set" );
		}
		else {

			m_classes = 1;

			bool present[ 2 ] = { false, false };
			for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

				if ( m_trainingLabels[ ii ] > 0 ) {

					m_trainingLabels[ ii ] = 1;
					present[ 1 ] = true;
				}
				else {

					m_trainingLabels[ ii ] = -1;
					present[ 0 ] = true;
				}
			}
			if ( ( ! present[ 0 ] ) || ( ! present[ 1 ] ) )
				throw std::runtime_error( "at least one positive and negative example must be present in training set" );
		}

		m_trainingVectorNormsSquared       = boost::shared_array< float >( new float[ m_rows ] );
		m_trainingVectorKernelNormsSquared = boost::shared_array< float >( new float[ m_rows ] );

		m_trainingResponses = boost::shared_array< double >( new double[ m_rows * m_classes ] );
		m_trainingAlphas = boost::shared_array< float >( new float[ m_rows * m_classes ] );

		ClusterTrainingVectors( smallClusters, activeClusters );

		Restart( regularization, kernel, kernelParameter1, kernelParameter2, kernelParameter3, biased );
	}
	catch( ... ) {

		Deinitialize();    // try to keep this structure in a valid state, if possible
		throw;
	}
}


void SVM::Load(
	char const* const filename,
	bool const smallClusters,
	unsigned int const activeClusters
)
{
	if ( ! m_constructed )
		throw std::runtime_error( "SVM has not been successfully constructed" );
	if ( m_initializedHost )
		Deinitialize();
	BOOST_ASSERT( ! m_initializedDevice );
	BOOST_ASSERT( ! m_initializedHost );
	m_initializedHost = true;

	try {

		{	FILE* file = fopen( filename, "rb" );
			if ( file == NULL )
				throw std::runtime_error( "Unable to open file" );

			if ( fread( &m_rows, sizeof( m_rows ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read rows" );
			if ( fread( &m_columns, sizeof( m_columns ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read columns" );
			if ( fread( &m_classes, sizeof( m_classes ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read classes" );

			m_trainingVectors = boost::shared_array< SparseVector >( new SparseVector[ m_rows ] );
			for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

				boost::uint32_t size;
				if ( fread( &size, sizeof( boost::uint32_t ), 1, file ) != 1 )
					throw std::runtime_error( "Unable to read training vector size" );

				for ( unsigned int jj = 0; jj < size; ++jj ) {

					boost::uint32_t index;
					float value;
					if ( fread( &index, sizeof( boost::uint32_t ), 1, file ) != 1 )
						throw std::runtime_error( "Unable to read training vector nonzero index" );
					if ( fread( &value, sizeof( float ), 1, file ) != 1 )
						throw std::runtime_error( "Unable to read training vector nonzero value" );

					m_trainingVectors[ ii ].push_back( std::pair< unsigned int, float >( index, value ) );
				}
			}

			m_trainingLabels = boost::shared_array< boost::int32_t >( new boost::int32_t[ m_rows ] );
			if ( fread( m_trainingLabels.get(), sizeof( boost::int32_t ), m_rows, file ) != m_rows )
				throw std::runtime_error( "Unable to read training labels" );

			m_trainingVectorNormsSquared       = boost::shared_array< float >( new float[ m_rows ] );
			m_trainingVectorKernelNormsSquared = boost::shared_array< float >( new float[ m_rows ] );

			m_trainingResponses = boost::shared_array< double >( new double[ m_rows * m_classes ] );
			m_trainingAlphas = boost::shared_array< float >( new float[ m_rows * m_classes ] );
			if ( fread( m_trainingResponses.get(), sizeof( double ), m_rows * m_classes, file ) != m_rows * m_classes )
				throw std::runtime_error( "Unable to read training responses" );
			if ( fread( m_trainingAlphas.get(), sizeof( float ), m_rows * m_classes, file ) != m_rows * m_classes )
				throw std::runtime_error( "Unable to read training alphas" );

			if ( fread( &m_regularization, sizeof( float ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read regularization parameter" );
			{	boost::int32_t kernel;
				if ( fread( &kernel, sizeof( boost::int32_t ), 1, file ) != 1 )
					throw std::runtime_error( "Unable to read kernel" );
				m_kernel = static_cast< GTSVM_Kernel >( kernel );
			}
			if ( fread( &m_kernelParameter1, sizeof( float ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read first kernel parameter" );
			if ( fread( &m_kernelParameter2, sizeof( float ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read second kernel parameter" );
			if ( fread( &m_kernelParameter3, sizeof( float ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read third kernel parameter" );
			if ( fread( &m_biased, sizeof( bool ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read biased flag" );
			if ( fread( &m_bias, sizeof( float ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to read bias" );

			fclose( file );
		}

		for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

			double accumulator = 0;

			SparseVector::const_iterator jj    = m_trainingVectors[ ii ].begin();
			SparseVector::const_iterator jjEnd = m_trainingVectors[ ii ].end();
			for ( ; jj != jjEnd; ++jj )
				accumulator += Square( jj->second );

			double value = std::numeric_limits< double >::quiet_NaN();
			switch( m_kernel ) {
				case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( accumulator, accumulator, accumulator, m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( accumulator, accumulator, accumulator, m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( accumulator, accumulator, accumulator, m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				default: throw std::runtime_error( "Unknown kernel" );
			}

			m_trainingVectorNormsSquared[       ii ] = accumulator;
			m_trainingVectorKernelNormsSquared[ ii ] = value;
		}

		ClusterTrainingVectors( smallClusters, activeClusters );
	}
	catch( ... ) {

		Deinitialize();    // try to keep this structure in a valid state, if possible
		throw;
	}
}


void SVM::Save( char const* const filename ) const {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	const_cast< SVM* >( this )->UpdateResponses();
	BOOST_ASSERT( m_updatedResponses );

	FILE* file = fopen( filename, "wb" );
	if ( file == NULL )
		throw std::runtime_error( "Unable to open file" );

	if ( fwrite( &m_rows, sizeof( m_rows ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write rows" );
	if ( fwrite( &m_columns, sizeof( m_columns ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write columns" );
	if ( fwrite( &m_classes, sizeof( m_classes ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write number of classes" );

	for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

		boost::uint32_t const size = m_trainingVectors[ ii ].size();
		if ( fwrite( &size, sizeof( boost::uint32_t ), 1, file ) != 1 )
			throw std::runtime_error( "Unable to write training vector size" );

		SparseVector::const_iterator jj    = m_trainingVectors[ ii ].begin();
		SparseVector::const_iterator jjEnd = m_trainingVectors[ ii ].end();
		for ( ; jj != jjEnd; ++jj ) {

			boost::uint32_t const index = jj->first;
			float const value = jj->second;
			if ( fwrite( &index, sizeof( boost::uint32_t ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to write training vector nonzero index" );
			if ( fwrite( &value, sizeof( float ), 1, file ) != 1 )
				throw std::runtime_error( "Unable to write training vector nonzero value" );
		}
	}

	if ( fwrite( m_trainingLabels.get(), sizeof( boost::int32_t ), m_rows, file ) != m_rows )
		throw std::runtime_error( "Unable to write training labels" );

	if ( fwrite( m_trainingResponses.get(), sizeof( double ), m_rows * m_classes, file ) != m_rows * m_classes )
		throw std::runtime_error( "Unable to write training responses" );
	if ( fwrite( m_trainingAlphas.get(), sizeof( float ), m_rows * m_classes, file ) != m_rows * m_classes )
		throw std::runtime_error( "Unable to write training alphas" );

	if ( fwrite( &m_regularization, sizeof( float ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write regularization parameter" );
	{	boost::int32_t const kernel = m_kernel;
		if ( fwrite( &kernel, sizeof( boost::int32_t ), 1, file ) != 1 )
			throw std::runtime_error( "Unable to write kernel" );
	}
	if ( fwrite( &m_kernelParameter1, sizeof( float ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write first kernel parameter" );
	if ( fwrite( &m_kernelParameter2, sizeof( float ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write second kernel parameter" );
	if ( fwrite( &m_kernelParameter3, sizeof( float ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write third kernel parameter" );
	if ( fwrite( &m_biased, sizeof( bool ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write biased flag" );
	if ( fwrite( &m_bias, sizeof( float ), 1, file ) != 1 )
		throw std::runtime_error( "Unable to write bias" );

	fclose( file );
}


void SVM::Shrink( bool const smallClusters, unsigned int const activeClusters ) {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );
	DeinitializeDevice();
	BOOST_ASSERT( m_updatedResponses );

	m_clusterIndices.clear();
	m_clusterNonzeroIndices.clear();

	unsigned int rows = 0;
	for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

		bool zero = true;
		for ( unsigned int jj = 0; jj < m_classes; ++jj ) {

			if ( m_trainingAlphas[ ii * m_classes + jj ] != 0 ) {

				zero = false;
				break;
			}
		}

		if ( ! zero )
			++rows;
	}

	boost::shared_array< SparseVector > trainingVectors( new SparseVector[ rows ] );
	boost::shared_array< boost::int32_t > trainingLabels( new boost::int32_t[ rows ] );
	boost::shared_array< float > trainingVectorNormsSquared( new float[ rows ] );
	boost::shared_array< float > trainingVectorKernelNormsSquared( new float[ rows ] );
	boost::shared_array< double > trainingResponses( new double[ rows * m_classes ] );
	boost::shared_array< float > trainingAlphas( new float[ rows * m_classes ] );

	{	unsigned int kk = 0;
		for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

			bool zero = true;
			for ( unsigned int jj = 0; jj < m_classes; ++jj ) {

				if ( m_trainingAlphas[ ii * m_classes + jj ] != 0 ) {

					zero = false;
					break;
				}
			}

			if ( ! zero ) {

				trainingVectors[                  kk ] = m_trainingVectors[                  ii ];
				trainingLabels[                   kk ] = m_trainingLabels[                   ii ];
				trainingVectorNormsSquared[       kk ] = m_trainingVectorNormsSquared[       ii ];
				trainingVectorKernelNormsSquared[ kk ] = m_trainingVectorKernelNormsSquared[ ii ];
				for ( unsigned int jj = 0; jj < m_classes; ++jj ) {

					if ( m_trainingAlphas[ ii * m_classes + jj ] != 0 ) {

						trainingResponses[ kk * m_classes + jj ] = m_trainingResponses[ ii * m_classes + jj ];
						trainingAlphas[    kk * m_classes + jj ] = m_trainingAlphas[    ii * m_classes + jj ];
					}
				}
				++kk;
			}
		}
		BOOST_ASSERT( kk == rows );
	}

	m_rows = rows;
	m_trainingVectors                  = trainingVectors;
	m_trainingLabels                   = trainingLabels;
	m_trainingVectorNormsSquared       = trainingVectorNormsSquared;
	m_trainingVectorKernelNormsSquared = trainingVectorKernelNormsSquared;
	m_trainingResponses                = trainingResponses;
	m_trainingAlphas                   = trainingAlphas;

	ClusterTrainingVectors( smallClusters, activeClusters );
}


void SVM::DeinitializeDevice() {

	if ( m_initializedDevice ) {

		UpdateResponses();
		BOOST_ASSERT( m_updatedResponses );

		BOOST_ASSERT( m_initializedHost );
		m_initializedDevice = false;

		for ( unsigned int ii = 0; ii < ARRAYLENGTH( m_deviceWork ); ++ii ) {

			if ( m_deviceWork[ ii ] != NULL ) {

				CUDA_VERIFY( "Failed to free work on device", cudaFree( m_deviceWork[ ii ] ) );
				m_deviceWork[ ii ] = NULL;
			}
		}

		if ( m_batchVectorsTranspose != NULL ) {

			CUDA_VERIFY( "Failed to free batch vectors on host", cudaFreeHost( m_batchVectorsTranspose ) );
			m_batchVectorsTranspose = NULL;
		}
		if ( m_deviceBatchVectorsTranspose != NULL ) {

			CUDA_VERIFY( "Failed to free batch vectors on device", cudaFree( m_deviceBatchVectorsTranspose ) );
			m_deviceBatchVectorsTranspose = NULL;
		}

		if ( m_batchResponses != NULL ) {

			CUDA_VERIFY( "Failed to free batch responses on host", cudaFreeHost( m_batchResponses ) );
			m_batchResponses = NULL;
		}
		if ( m_deviceBatchResponses != NULL ) {

			CUDA_VERIFY( "Failed to free batch responses on device", cudaFree( m_deviceBatchResponses ) );
			m_deviceBatchResponses = NULL;
		}

		if ( m_batchAlphas != NULL ) {

			CUDA_VERIFY( "Failed to free batch alphas on host", cudaFreeHost( m_batchAlphas ) );
			m_batchAlphas = NULL;
		}
		if ( m_deviceBatchAlphas != NULL ) {

			CUDA_VERIFY( "Failed to free batch alphas on device", cudaFree( m_deviceBatchAlphas ) );
			m_deviceBatchAlphas = NULL;
		}

		if ( m_batchIndices != NULL ) {

			CUDA_VERIFY( "Failed to free batch indices on host", cudaFreeHost( m_batchIndices ) );
			m_batchIndices = NULL;
		}
		if ( m_deviceBatchIndices != NULL ) {

			CUDA_VERIFY( "Failed to free batch indices on device", cudaFree( m_deviceBatchIndices ) );
			m_deviceBatchIndices = NULL;
		}

		if ( m_deviceTrainingLabels != NULL ) {

			CUDA_VERIFY( "Failed to free training labels on device", cudaFree( m_deviceTrainingLabels ) );
			m_deviceTrainingLabels = NULL;
		}
		if ( m_deviceTrainingVectorNormsSquared != NULL ) {

			CUDA_VERIFY( "Failed to free training vector squared norms on device", cudaFree( m_deviceTrainingVectorNormsSquared ) );
			m_deviceTrainingVectorNormsSquared = NULL;
		}
		if ( m_deviceTrainingVectorKernelNormsSquared != NULL ) {

			CUDA_VERIFY( "Failed to free training vector kernel squared norms on device", cudaFree( m_deviceTrainingVectorKernelNormsSquared ) );
			m_deviceTrainingVectorKernelNormsSquared = NULL;
		}
		if ( m_deviceTrainingResponses != NULL ) {

			CUDA_VERIFY( "Failed to free training responses on device", cudaFree( m_deviceTrainingResponses ) );
			m_deviceTrainingResponses = NULL;
		}
		if ( m_deviceTrainingAlphas != NULL ) {

			CUDA_VERIFY( "Failed to free training alphas on device", cudaFree( m_deviceTrainingAlphas ) );
			m_deviceTrainingAlphas = NULL;
		}
		if ( m_deviceNonzeroIndices != NULL ) {

			CUDA_VERIFY( "Failed to nonzero indices on device", cudaFree( m_deviceNonzeroIndices ) );
			m_deviceNonzeroIndices = NULL;
		}
		if ( m_deviceTrainingVectorsTranspose != NULL ) {

			CUDA_VERIFY( "Failed to free training vectors on device", cudaFree( m_deviceTrainingVectorsTranspose ) );
			m_deviceTrainingVectorsTranspose = NULL;
		}
		if ( m_deviceClusterHeaders != NULL ) {

			CUDA_VERIFY( "Failed to free cluster headers on device", cudaFree( m_deviceClusterHeaders ) );
			m_deviceTrainingAlphas = NULL;
		}
		if ( m_deviceClusterSizeSums != NULL ) {

			CUDA_VERIFY( "Failed to free cluster size sums on device", cudaFree( m_deviceClusterSizeSums ) );
			m_deviceTrainingAlphas = NULL;
		}
	}
}


void SVM::Deinitialize() {

	DeinitializeDevice();
	BOOST_ASSERT( ! m_initializedDevice );

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );
	m_initializedHost = false;

	m_trainingVectors = boost::shared_array< SparseVector >();
	m_trainingLabels = boost::shared_array< boost::int32_t >();
	m_trainingVectorNormsSquared = boost::shared_array< float >();
	m_trainingVectorKernelNormsSquared = boost::shared_array< float >();

	m_trainingResponses = boost::shared_array< double >();
	m_trainingAlphas = boost::shared_array< float >();

	m_clusterIndices.clear();
	m_clusterNonzeroIndices.clear();
}


void SVM::GetTrainingVectorsSparse(
	void* const trainingVectors,    // order depends on the columnMajor flag
	size_t* const trainingVectorIndices,
	size_t* const trainingVectorOffsets,
	GTSVM_Type trainingVectorsType,
	bool const columnMajor
) const
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	SVM_SparseSparseReverseMemcpy2d(
		trainingVectors,
		trainingVectorIndices,
		trainingVectorOffsets,
		trainingVectorsType,
		m_trainingVectors.get(),
		m_rows,
		m_columns,
		columnMajor
	);
}


void SVM::GetTrainingVectorsDense(
	void* const trainingVectors,    // order depends on the columnMajor flag
	GTSVM_Type trainingVectorsType,
	bool const columnMajor
) const
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	SVM_SparseReverseMemcpy2d(
		trainingVectors,
		trainingVectorsType,
		m_trainingVectors.get(),
		m_rows,
		m_columns,
		columnMajor
	);
}


void SVM::GetTrainingLabels(
	void* const trainingLabels,
	GTSVM_Type trainingLabelsType
) const
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	SVM_ReverseMemcpy(
		trainingLabels,
		0,
		trainingLabelsType,
		m_trainingLabels.get(),
		m_rows
	);
}


void SVM::GetTrainingResponses(
	void* const trainingResponses,
	GTSVM_Type trainingResponsesType,
	bool const columnMajor
) const
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	const_cast< SVM* >( this )->UpdateResponses();
	BOOST_ASSERT( m_updatedResponses );

	SVM_ReverseMemcpy2d(
		trainingResponses,
		trainingResponsesType,
		m_trainingResponses.get(),
		m_rows,
		m_classes,
		columnMajor
	);
}


void SVM::GetAlphas(
	void* const trainingAlphas,
	GTSVM_Type trainingAlphasType,
	bool const columnMajor
) const
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	SVM_ReverseMemcpy2d(
		trainingAlphas,
		trainingAlphasType,
		m_trainingAlphas.get(),
		m_rows,
		m_classes,
		columnMajor
	);
}


void SVM::SetAlphas(
	void const* const trainingAlphas,
	GTSVM_Type trainingAlphasType,
	bool const columnMajor
)
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );
	DeinitializeDevice();

	SVM_Memcpy2d(
		m_trainingAlphas.get(),
		trainingAlphas,
		trainingAlphasType,
		m_rows,
		m_classes,
		columnMajor
	);

	Recalculate();
}


void SVM::Recalculate() {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );
	if ( ! m_initializedDevice )
		InitializeDevice();
	BOOST_ASSERT( m_initializedDevice );

	for ( unsigned int ii = 0; ii < m_rows; ii += 16 ) {

		unsigned int const batchSize = std::min( 16u, m_rows - ii );

		for ( unsigned int jj = 0; jj < batchSize; ++jj ) {

			unsigned int ll = 0;
			SparseVector::const_iterator kk    = m_trainingVectors[ ii + jj ].begin();
			SparseVector::const_iterator kkEnd = m_trainingVectors[ ii + jj ].end();
			for ( ; kk != kkEnd; ++kk ) {

				BOOST_ASSERT( kk->first < m_columns );
				for ( ; ll < kk->first; ++ll )
					m_batchVectorsTranspose[ ll * 16 + jj ] = 0;
				m_batchVectorsTranspose[ ll * 16 + jj ] = kk->second;
				++ll;
			}
			for ( ; ll < m_columns; ++ll )
				m_batchVectorsTranspose[ ll * 16 + jj ] = 0;

			m_batchVectorNormsSquared[ jj ] = m_trainingVectorNormsSquared[ ii + jj ];
		}

		CUDA_VERIFY(
			"Failed to copy batch to device",
			cudaMemcpy(
				m_deviceBatchVectorsTranspose,
				m_batchVectorsTranspose,
				( m_columns << 4 ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch squared norms to device",
			cudaMemcpy(
				m_deviceBatchVectorNormsSquared,
				m_batchVectorNormsSquared,
				batchSize * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_FLOAT_DOUBLE const* const deviceResult = CUDA::SparseEvaluateKernel(
			m_deviceWork[ 0 ],
			m_deviceWork[ 1 ],
			m_deviceBatchVectorsTranspose,
			m_deviceBatchVectorNormsSquared,
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			m_classes,
			m_workSize,
			m_kernel,
			m_kernelParameter1,
			m_kernelParameter2,
			m_kernelParameter3
		);

		CUDA_VERIFY(
			"Failed to copy responses from device",
			cudaMemcpy(
				m_batchResponses,
				deviceResult,
				batchSize * m_classes * sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyDeviceToHost
			)
		);

		for ( unsigned int jj = 0; jj < batchSize; ++jj )
			for ( unsigned int kk = 0; kk < m_classes; ++kk )
				m_trainingResponses[ ( ii + jj ) * m_classes + kk ] = m_batchResponses[ kk * 16 + jj ];
	}

	{	CUDA_FLOAT_DOUBLE* trainingResponses;
		CUDA_VERIFY(
			"Failed to allocate space for training responses on host",
			cudaMallocHost( &trainingResponses, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ) )
		);

		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();
			for ( unsigned int jj = 0; jj < size; ++jj )
				for ( unsigned int kk = 0; kk < m_classes; ++kk )
					trainingResponses[ ( ( ii * m_classes + kk ) << m_logMaximumClusterSize ) + jj ] = m_trainingResponses[ m_clusterIndices[ ii ][ jj ] * m_classes + kk ];
			for ( unsigned int jj = size; jj < ( 1u << m_logMaximumClusterSize ); ++jj )
				for ( unsigned int kk = 0; kk < m_classes; ++kk )
					trainingResponses[ ( ( ii * m_classes + kk ) << m_logMaximumClusterSize ) + jj ] = 0;
		}
		CUDA_VERIFY(
			"Failed to copy training responses to device",
			cudaMemcpy(
				m_deviceTrainingResponses,
				trainingResponses,
				( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY( "Failed to free training responses on host", cudaFreeHost( trainingResponses ) );
	}
	m_updatedResponses = true;

	if ( m_biased ) {

		if ( m_classes != 1 )
			throw std::runtime_error( "Multiclass is only implemented for problems without an unregularized bias" );

		CUDA_FLOAT_DOUBLE numerator = 0;
		boost::uint32_t denominator = 0;
		std::pair< CUDA_FLOAT_DOUBLE const*, boost::uint32_t const* > const deviceResult = CUDA::SparseCalculateBias(
			m_deviceWork[ 0 ],
			m_deviceWork[ 1 ],
			m_deviceWork[ 2 ],
			m_deviceWork[ 3 ],
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			m_workSize,
			m_regularization
		);
		CUDA_VERIFY(
			"Failed to copy bias numerator from device",
			cudaMemcpy(
				&numerator,
				deviceResult.first,
				sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyDeviceToHost
			)
		);
		CUDA_VERIFY(
			"Failed to copy bias denominator from device",
			cudaMemcpy(
				&denominator,
				deviceResult.second,
				sizeof( boost::uint32_t ),
				cudaMemcpyDeviceToHost
			)
		);

		m_bias = ( ( denominator != 0 ) ? ( numerator / denominator ) : 0 );
	}
}


void SVM::Restart(
	float const regularization,
	GTSVM_Kernel const kernel,
	float const kernelParameter1,
	float const kernelParameter2,
	float const kernelParameter3,
	bool const biased
)
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );

	if ( boost::math::isinf( regularization ) )
		throw std::runtime_error( "The regularization parameter must be finite" );
	if ( boost::math::isnan( regularization ) )
		throw std::runtime_error( "The regularization parameter cannot be NaN" );
	if ( regularization <= 0 )
		throw std::runtime_error( "The regularization parameter must be positive" );

	switch( kernel ) {

		case GTSVM_KERNEL_GAUSSIAN: {

			if ( boost::math::isinf( kernelParameter1 ) )
				throw std::runtime_error( "The first kernel parameter must be finite" );
			if ( boost::math::isnan( kernelParameter1 ) )
				throw std::runtime_error( "The first kernel parameter cannot be NaN" );
			if ( kernelParameter1 <= 0 )
				throw std::runtime_error( "The first kernel parameter must be positive" );
			break;
		}

		case GTSVM_KERNEL_POLYNOMIAL: {

			if ( boost::math::isinf( kernelParameter1 ) )
				throw std::runtime_error( "The first kernel parameter must be finite" );
			if ( boost::math::isnan( kernelParameter1 ) )
				throw std::runtime_error( "The first kernel parameter cannot be NaN" );
			if ( boost::math::isinf( kernelParameter2 ) )
				throw std::runtime_error( "The second kernel parameter must be finite" );
			if ( boost::math::isnan( kernelParameter2 ) )
				throw std::runtime_error( "The second kernel parameter cannot be NaN" );
			if ( boost::math::isinf( kernelParameter3 ) )
				throw std::runtime_error( "The third kernel parameter must be finite" );
			if ( boost::math::isnan( kernelParameter3 ) )
				throw std::runtime_error( "The third kernel parameter cannot be NaN" );
			if ( kernelParameter3 <= 0 )
				throw std::runtime_error( "The third kernel parameter must be positive" );
			break;
		}

		case GTSVM_KERNEL_SIGMOID: {

			if ( boost::math::isinf( kernelParameter1 ) )
				throw std::runtime_error( "The first kernel parameter must be finite" );
			if ( boost::math::isnan( kernelParameter1 ) )
				throw std::runtime_error( "The first kernel parameter cannot be NaN" );
			if ( boost::math::isinf( kernelParameter2 ) )
				throw std::runtime_error( "The second kernel parameter must be finite" );
			if ( boost::math::isnan( kernelParameter2 ) )
				throw std::runtime_error( "The second kernel parameter cannot be NaN" );
			break;
		}

		default: throw std::runtime_error( "Unknown kernel" );
	}

	m_regularization = regularization;
	m_kernel = kernel;
	m_kernelParameter1 = kernelParameter1;
	m_kernelParameter2 = kernelParameter2;
	m_kernelParameter3 = kernelParameter3;
	m_biased = biased;

	m_bias = 0;

	for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

		double accumulator = 0;

		SparseVector::const_iterator jj    = m_trainingVectors[ ii ].begin();
		SparseVector::const_iterator jjEnd = m_trainingVectors[ ii ].end();
		for ( ; jj != jjEnd; ++jj )
			accumulator += Square( jj->second );

		double value = std::numeric_limits< double >::quiet_NaN();
		switch( m_kernel ) {
			case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( accumulator, accumulator, accumulator, m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( accumulator, accumulator, accumulator, m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( accumulator, accumulator, accumulator, m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			default: throw std::runtime_error( "Unknown kernel" );
		}

		m_trainingVectorNormsSquared[       ii ] = accumulator;
		m_trainingVectorKernelNormsSquared[ ii ] = value;
	}

	std::fill( m_trainingResponses.get(), m_trainingResponses.get() + m_rows * m_classes, 0 );
	if ( m_initializedDevice ) {

#if 0
		// CUDA sometimes thinks that we're trying to clear too large a buffer
		CUDA_VERIFY(
			"Failed to clear training responses on device",
			cudaMemset( &m_deviceTrainingResponses, 0, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ) )
		);
#else    // 0/1
		CUDA_FLOAT_DOUBLE* trainingResponses;
		CUDA_VERIFY(
			"Failed to allocate space for training responses on host",
			cudaMallocHost( &trainingResponses, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ) )
		);
		std::fill( trainingResponses, trainingResponses + ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ), 0.0f );
		CUDA_VERIFY(
			"Failed to copy training responses to device",
			cudaMemcpy(
				m_deviceTrainingResponses,
				trainingResponses,
				( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyHostToDevice
			)
		);
		CUDA_VERIFY( "Failed to free training responses on host", cudaFreeHost( trainingResponses ) );
#endif    // 0/1
	}
	m_updatedResponses = true;

	std::fill( m_trainingAlphas.get(), m_trainingAlphas.get() + m_rows * m_classes, 0.0f );
	if ( m_initializedDevice ) {

#if 0
		// CUDA sometimes thinks that we're trying to clear too large a buffer
		CUDA_VERIFY(
			"Failed to clear training alphas on device",
			cudaMemset( &m_deviceTrainingAlphas, 0, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( float ) )
		);
#else    // 0/1
		float* trainingAlphas;
		CUDA_VERIFY(
			"Failed to allocate space for training alphas on host",
			cudaMallocHost( &trainingAlphas, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( float ) )
		);
		std::fill( trainingAlphas, trainingAlphas + ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ), 0.0f );
		CUDA_VERIFY(
			"Failed to copy training alphas to device",
			cudaMemcpy(
				m_deviceTrainingAlphas,
				trainingAlphas,
				( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);
		CUDA_VERIFY( "Failed to free training alphas on host", cudaFreeHost( trainingAlphas ) );
#endif    // 0/1
	}
}


std::pair< CUDA_FLOAT_DOUBLE, CUDA_FLOAT_DOUBLE > const SVM::Optimize( unsigned int const iterations ) {

	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );
	if ( ! m_initializedDevice )
		InitializeDevice();
	BOOST_ASSERT( m_initializedDevice );

	bool progress = false;
	if ( m_biased ) {

		if ( m_classes != 1 )
			throw std::runtime_error( "Multiclass is only implemented for problems without an unregularized bias" );

		progress = true;
		for ( unsigned int ii = 0; progress && ( ii < iterations ); ii += 16 )
			progress = IterateBiasedBinary();

		CUDA_FLOAT_DOUBLE numerator = 0;
		boost::uint32_t denominator = 0;
		std::pair< CUDA_FLOAT_DOUBLE const*, boost::uint32_t const* > const deviceResult = CUDA::SparseCalculateBias(
			m_deviceWork[ 0 ],
			m_deviceWork[ 1 ],
			m_deviceWork[ 2 ],
			m_deviceWork[ 3 ],
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			m_workSize,
			m_regularization
		);
		CUDA_VERIFY(
			"Failed to copy bias numerator from device",
			cudaMemcpy(
				&numerator,
				deviceResult.first,
				sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyDeviceToHost
			)
		);
		CUDA_VERIFY(
			"Failed to copy bias denominator from device",
			cudaMemcpy(
				&denominator,
				deviceResult.second,
				sizeof( boost::uint32_t ),
				cudaMemcpyDeviceToHost
			)
		);

		m_bias = ( ( denominator != 0 ) ? ( numerator / denominator ) : 0 );
	}
	else {

		if ( m_classes == 1 ) {

			progress = true;
			for ( unsigned int ii = 0; progress && ( ii < iterations ); ii += 16 )
				progress = IterateUnbiasedBinary();

			BOOST_ASSERT( m_bias == 0 );
		}
		else {

			progress = true;
			for ( unsigned int ii = 0; progress && ( ii < iterations ); ii += 16 )
				progress = IterateUnbiasedMulticlass();
		}
	}

	CUDA_FLOAT_DOUBLE primal =  std::numeric_limits< CUDA_FLOAT_DOUBLE >::infinity();
	CUDA_FLOAT_DOUBLE dual   = -std::numeric_limits< CUDA_FLOAT_DOUBLE >::infinity();

	std::pair< CUDA_FLOAT_DOUBLE const*, CUDA_FLOAT_DOUBLE const* > const deviceResult = CUDA::SparseCalculateObjectives(
		m_deviceWork[ 0 ],
		m_deviceWork[ 1 ],
		m_deviceWork[ 2 ],
		m_deviceWork[ 3 ],
		m_deviceClusterHeaders,
		m_logMaximumClusterSize,
		m_clusters,
		m_classes,
		m_workSize,
		m_regularization,
		m_bias
	);
	CUDA_VERIFY(
		"Failed to copy primal objective value from device",
		cudaMemcpy(
			&primal,
			deviceResult.first,
			sizeof( CUDA_FLOAT_DOUBLE ),
			cudaMemcpyDeviceToHost
		)
	);
	CUDA_VERIFY(
		"Failed to copy dual objective value from device",
		cudaMemcpy(
			&dual,
			deviceResult.second,
			sizeof( CUDA_FLOAT_DOUBLE ),
			cudaMemcpyDeviceToHost
		)
	);

	m_updatedResponses = false;

	if ( ! progress )
		throw std::runtime_error( "An iteration made no progress" );

	return std::pair< CUDA_FLOAT_DOUBLE, CUDA_FLOAT_DOUBLE >( primal, dual );
}


void SVM::ClassifySparse(
	void* const result,
	GTSVM_Type resultType,
	void const* const vectors,    // order depends on columnMajor flag
	size_t const* const vectorIndices,
	size_t const* const vectorOffsets,
	GTSVM_Type vectorsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor
)
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );
	if ( ! m_initializedDevice )
		InitializeDevice();
	BOOST_ASSERT( m_initializedDevice );

	// **TODO: it would be nice to not copy all of this
	boost::shared_array< SparseVector > sparseVectors( new SparseVector[ rows ] );
	SVM_SparseSparseMemcpy2d( sparseVectors.get(), vectors, vectorIndices, vectorOffsets, vectorsType, rows, columns, columnMajor );

	// **TODO: it would be nice to not copy all of this
	boost::shared_array< CUDA_FLOAT_DOUBLE > classifications( new CUDA_FLOAT_DOUBLE[ rows * m_classes ] );

	for ( unsigned int ii = 0; ii < rows; ii += 16 ) {

		unsigned int const batchSize = std::min( 16u, rows - ii );

		for ( unsigned int jj = 0; jj < batchSize; ++jj ) {

			double accumulator = 0;

			unsigned int ll = 0;
			SparseVector::const_iterator kk    = sparseVectors[ ii + jj ].begin();
			SparseVector::const_iterator kkEnd = sparseVectors[ ii + jj ].end();
			for ( ; ( kk != kkEnd ) && ( kk->first < m_columns ); ++kk ) {

				for ( ; ll < kk->first; ++ll )
					m_batchVectorsTranspose[ ll * 16 + jj ] = 0;
				m_batchVectorsTranspose[ ll * 16 + jj ] = kk->second;
				accumulator += Square( kk->second );
				++ll;
			}
			for ( ; ll < m_columns; ++ll )
				m_batchVectorsTranspose[ ll * 16 + jj ] = 0;

			m_batchVectorNormsSquared[ jj ] = accumulator;
		}

		CUDA_VERIFY(
			"Failed to copy batch to device",
			cudaMemcpy(
				m_deviceBatchVectorsTranspose,
				m_batchVectorsTranspose,
				( m_columns << 4 ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch squared norms to device",
			cudaMemcpy(
				m_deviceBatchVectorNormsSquared,
				m_batchVectorNormsSquared,
				batchSize * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_FLOAT_DOUBLE const* const deviceResult = CUDA::SparseEvaluateKernel(
			m_deviceWork[ 0 ],
			m_deviceWork[ 1 ],
			m_deviceBatchVectorsTranspose,
			m_deviceBatchVectorNormsSquared,
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			m_classes,
			m_workSize,
			m_kernel,
			m_kernelParameter1,
			m_kernelParameter2,
			m_kernelParameter3
		);

		CUDA_VERIFY(
			"Failed to copy classifications from device",
			cudaMemcpy(
				m_batchResponses,
				deviceResult,
				batchSize * m_classes * sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyDeviceToHost
			)
		);

		for ( unsigned int jj = 0; jj < batchSize; ++jj )
			for ( unsigned int kk = 0; kk < m_classes; ++kk )
				classifications[ ( ii + jj ) * m_classes + kk ] = m_batchResponses[ kk * 16 + jj ];
	}

	if ( m_biased ) {

		CUDA_FLOAT_DOUBLE* ii    = classifications.get();
		CUDA_FLOAT_DOUBLE* iiEnd = ii + rows * m_classes;
		for ( ; ii != iiEnd; ++ii )
			*ii += m_bias;
	}

	SVM_ReverseMemcpy2d( result, resultType, classifications.get(), rows, m_classes, columnMajor );
}


void SVM::ClassifyDense(
	void* const result,
	GTSVM_Type resultType,
	void const* const vectors,    // order depends on columnMajor flag
	GTSVM_Type vectorsType,
	unsigned int const rows,
	unsigned int const columns,
	bool const columnMajor
)
{
	if ( ! m_initializedHost )
		throw std::runtime_error( "SVM has not been initialized" );
	if ( ! m_initializedDevice )
		InitializeDevice();
	BOOST_ASSERT( m_initializedDevice );

	// **TODO: it would be nice to not copy all of this
	boost::shared_array< CUDA_FLOAT_DOUBLE > classifications( new CUDA_FLOAT_DOUBLE[ rows * m_classes ] );

	for ( unsigned int ii = 0; ii < rows; ii += 16 ) {

		unsigned int const batchSize = std::min( 16u, rows - ii );

		for ( unsigned int jj = 0; jj < batchSize; ++jj ) {

			if ( columnMajor )
				SVM_MemcpyStride( m_batchVectorsTranspose + jj, 16, vectors, ii + jj, rows, vectorsType, std::min( columns, m_columns ) );
			else
				SVM_MemcpyStride( m_batchVectorsTranspose + jj, 16, vectors, ( ii + jj ) * m_columns, 1, vectorsType, std::min( columns, m_columns ) );
			for ( unsigned int kk = columns; kk < m_columns; ++kk )
				m_batchVectorsTranspose[ ( kk << 4 ) + jj ] = 0;

			double accumulator = 0;
			for ( unsigned int kk = 0; kk < m_columns; ++kk )
				accumulator += Square( m_batchVectorsTranspose[ ( kk << 4 ) + jj ] );
			m_batchVectorNormsSquared[ jj ] = accumulator;
		}

		CUDA_VERIFY(
			"Failed to copy batch to device",
			cudaMemcpy(
				m_deviceBatchVectorsTranspose,
				m_batchVectorsTranspose,
				( m_columns << 4 ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch squared norms to device",
			cudaMemcpy(
				m_deviceBatchVectorNormsSquared,
				m_batchVectorNormsSquared,
				batchSize * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_FLOAT_DOUBLE const* const deviceResult = CUDA::SparseEvaluateKernel(
			m_deviceWork[ 0 ],
			m_deviceWork[ 1 ],
			m_deviceBatchVectorsTranspose,
			m_deviceBatchVectorNormsSquared,
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			m_classes,
			m_workSize,
			m_kernel,
			m_kernelParameter1,
			m_kernelParameter2,
			m_kernelParameter3
		);

		CUDA_VERIFY(
			"Failed to copy classifications from device",
			cudaMemcpy(
				m_batchResponses,
				deviceResult,
				batchSize * m_classes * sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyDeviceToHost
			)
		);

		for ( unsigned int jj = 0; jj < batchSize; ++jj )
			for ( unsigned int kk = 0; kk < m_classes; ++kk )
				classifications[ ( ii + jj ) * m_classes + kk ] = m_batchResponses[ kk * 16 + jj ];
	}

	if ( m_biased ) {

		CUDA_FLOAT_DOUBLE* ii    = classifications.get();
		CUDA_FLOAT_DOUBLE* iiEnd = ii + rows * m_classes;
		for ( ; ii != iiEnd; ++ii )
			*ii += m_bias;
	}

	SVM_ReverseMemcpy2d( result, resultType, classifications.get(), rows, m_classes, columnMajor );
}


void SVM::Cleanup() {

	if ( ! m_constructed )
		throw std::runtime_error( "SVM has not been successfully constructed" );
	m_constructed = false;

	if ( m_initializedHost )
		Deinitialize();
	BOOST_ASSERT( ! m_initializedDevice );
	BOOST_ASSERT( ! m_initializedHost );

	if ( m_foundKeys != NULL ) {

		CUDA_VERIFY( "Failed to free found keys on host", cudaFreeHost( m_foundKeys ) );
		m_foundKeys = NULL;
	}
	if ( m_foundValues != NULL ) {

		CUDA_VERIFY( "Failed to free found values on host", cudaFreeHost( m_foundValues ) );
		m_foundValues = NULL;
	}

	if ( m_batchVectorNormsSquared != NULL ) {

		CUDA_VERIFY( "Failed to free batch squared norms on host", cudaFreeHost( m_batchVectorNormsSquared ) );
		m_batchVectorNormsSquared = NULL;
	}
	if ( m_deviceBatchVectorNormsSquared != NULL ) {

		CUDA_VERIFY( "Failed to free batch squared norms on device", cudaFree( m_deviceBatchVectorNormsSquared ) );
		m_deviceBatchVectorNormsSquared = NULL;
	}
}


void SVM::ClusterTrainingVectors(
	bool const smallClusters,
	unsigned int activeClusters
)
{
	m_logMaximumClusterSize = ( smallClusters ? 4 : 8 );

	unsigned int const densitySize = ( m_columns + ( 8 * sizeof( unsigned int ) - 1 ) ) / ( 8 * sizeof( unsigned int ) );

	m_clusterIndices.clear();
	m_clusterNonzeroIndices.clear();
	m_clusters = ( ( m_rows + ( ( 1u << m_logMaximumClusterSize ) - 1 ) ) >> m_logMaximumClusterSize );

	activeClusters = std::min( activeClusters, m_clusters );
	BOOST_ASSERT( activeClusters > 0 );

	boost::shared_array< std::vector< unsigned int > > clusterIndices( new std::vector< unsigned int >[ activeClusters ] );
	boost::shared_array< boost::shared_array< unsigned int > > clusterDensities( new boost::shared_array< unsigned int >[ activeClusters ] );
	for ( unsigned int ii = 0; ii < activeClusters; ++ii ) {

		boost::shared_array< unsigned int > clusterDensity( new unsigned int[ densitySize ] );
		std::fill( clusterDensity.get(), clusterDensity.get() + densitySize, 0 );
		clusterDensities[ ii ] = clusterDensity;
	}

	unsigned int remainingClusters = m_clusters;
	unsigned int currentClusters = std::min( remainingClusters, activeClusters );
	remainingClusters -= currentClusters;

	boost::shared_array< unsigned int > density( new unsigned int[ densitySize ] );

	boost::shared_array< unsigned int > indices( new unsigned int[ m_rows ] );
	for ( unsigned int ii = 0; ii < m_rows; ++ii )
		indices[ ii ] = ii;
	for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

		unsigned int jj = ( rand() % ( m_rows - ii ) ) + ii;
		if ( ii != jj )
			std::swap( indices[ ii ], indices[ jj ] );
	}

	for ( unsigned int ii = 0; ii < m_rows; ++ii ) {

		unsigned int const index = indices[ ii ];

		BOOST_ASSERT( currentClusters > 0 );

		std::fill( density.get(), density.get() + densitySize, 0 );

		{	SparseVector::const_iterator jj    = m_trainingVectors[ index ].begin();
			SparseVector::const_iterator jjEnd = m_trainingVectors[ index ].end();
			for ( ; jj != jjEnd; ++jj )
				density[ jj->first / ( 8 * sizeof( unsigned int ) ) ] |= ( 1u << ( jj->first & ( 8 * sizeof( unsigned int ) - 1 ) ) );
		}

		unsigned int clusterIndex = 0;
		unsigned int minimumCost = static_cast< unsigned int >( -1 );
		for ( unsigned int jj = 0; jj < currentClusters; ++jj ) {

			unsigned int newNonzeros = 0;
			unsigned int oldNonzeros = 0;

			for ( unsigned int kk = 0; kk < densitySize; ++kk ) {

				unsigned int newDensity = density[ kk ];
				unsigned int oldDensity = clusterDensities[ jj ][ kk ];

				newNonzeros += CountBits( ( newDensity ^ oldDensity ) & ~oldDensity );
				oldNonzeros += CountBits( ( newDensity ^ oldDensity ) & ~newDensity );
			}

			unsigned int const cost = oldNonzeros + newNonzeros * clusterIndices[ jj ].size();
			if ( cost < minimumCost ) {

				clusterIndex = jj;
				minimumCost = cost;
			}
		}

		clusterIndices[ clusterIndex ].push_back( index );
		for ( unsigned int kk = 0; kk < densitySize; ++kk )
			clusterDensities[ clusterIndex ][ kk ] |= density[ kk ];

		if ( clusterIndices[ clusterIndex ].size() >= ( 1u << m_logMaximumClusterSize ) ) {

			m_clusterIndices.push_back( clusterIndices[ clusterIndex ] );
			{	std::vector< unsigned int > nonzeros;
				for ( unsigned int jj = 0; jj < densitySize; ++jj ) {

					for ( unsigned int kk = 0; kk < 8 * sizeof( unsigned int ); ++kk ) {

						if ( clusterDensities[ clusterIndex ][ jj ] & ( 1u << kk ) )
							nonzeros.push_back( jj * 8 * sizeof( unsigned int ) + kk );
					}
				}
				m_clusterNonzeroIndices.push_back( nonzeros );
			}

			clusterIndices[ clusterIndex ].clear();
			std::fill( clusterDensities[ clusterIndex ].get(), clusterDensities[ clusterIndex ].get() + densitySize, 0 );
			--currentClusters;
			clusterIndices[ clusterIndex ].swap( clusterIndices[ currentClusters ] );
			std::swap( clusterDensities[ clusterIndex ], clusterDensities[ currentClusters ] );

			if ( remainingClusters > 0 ) {

				++currentClusters;
				--remainingClusters;
			}
		}
	}
	BOOST_ASSERT( remainingClusters == 0 );

	for ( unsigned int ii = 0; ii < currentClusters; ++ii ) {

		if ( clusterIndices[ ii ].size() > 0 ) {

			m_clusterIndices.push_back( clusterIndices[ ii ] );
			{	std::vector< unsigned int > nonzeros;
				for ( unsigned int jj = 0; jj < densitySize; ++jj ) {

					for ( unsigned int kk = 0; kk < 8 * sizeof( unsigned int ); ++kk ) {

						if ( clusterDensities[ ii ][ jj ] & ( 1u << kk ) )
							nonzeros.push_back( jj * 8 * sizeof( unsigned int ) + kk );
					}
				}
				m_clusterNonzeroIndices.push_back( nonzeros );
			}
		}
	}
	BOOST_ASSERT( m_clusterIndices.size()        == m_clusters );
	BOOST_ASSERT( m_clusterNonzeroIndices.size() == m_clusters );
}


void SVM::InitializeDevice() {

	if ( m_initializedDevice )
		throw std::runtime_error( "SVM has already been initialized" );
	m_initializedDevice = true;

	CUDA_VERIFY(
		"Failed to allocate space for batch vectors on host",
		cudaMallocHost( &m_batchVectorsTranspose, ( m_columns << 4 ) * sizeof( float ) )
	);
	CUDA_VERIFY(
		"Failed to allocate space for batch vectors on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceBatchVectorsTranspose ), ( m_columns << 4 ) * sizeof( float ) )
	);

	CUDA_VERIFY(
		"Failed to allocate space for batch responses on host",
		cudaMallocHost( &m_batchResponses, 16 * m_classes * sizeof( CUDA_FLOAT_DOUBLE ) )
	);
	CUDA_VERIFY(
		"Failed to allocate space for batch responses on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceBatchResponses ), 16 * m_classes * sizeof( CUDA_FLOAT_DOUBLE ) )
	);

	CUDA_VERIFY(
		"Failed to allocate space for batch alphas on host",
		cudaMallocHost( &m_batchAlphas, 16 * m_classes * sizeof( float ) )
	);
	CUDA_VERIFY(
		"Failed to allocate space for batch alphas on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceBatchAlphas ), 16 * m_classes * sizeof( float ) )
	);

	CUDA_VERIFY(
		"Failed to allocate space for batch indices on host",
		cudaMallocHost( &m_batchIndices, 16 * m_classes * sizeof( boost::uint32_t ) )
	);
	CUDA_VERIFY(
		"Failed to allocate space for batch indices on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceBatchIndices ), 16 * m_classes * sizeof( boost::uint32_t ) )
	);

	CUDA_VERIFY(
		"Failed to allocate space for training labels on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceTrainingLabels ), ( m_clusters << m_logMaximumClusterSize ) * sizeof( boost::int32_t ) )
	);
	{	boost::int32_t* trainingLabels;
		CUDA_VERIFY(
			"Failed to allocate space for training labels on host",
			cudaMallocHost( &trainingLabels, ( m_clusters << m_logMaximumClusterSize ) * sizeof( boost::int32_t ) )
		);

		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();
			for ( unsigned int jj = 0; jj < size; ++jj )
				trainingLabels[ ( ii << m_logMaximumClusterSize ) + jj ] = m_trainingLabels[ m_clusterIndices[ ii ][ jj ] ];
			for ( unsigned int jj = size; jj < ( 1u << m_logMaximumClusterSize ); ++jj )
				trainingLabels[ ( ii << m_logMaximumClusterSize ) + jj ] = 0;
		}
		CUDA_VERIFY(
			"Failed to copy training labels to device",
			cudaMemcpy(
				m_deviceTrainingLabels,
				trainingLabels,
				( m_clusters << m_logMaximumClusterSize ) * sizeof( boost::int32_t ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY( "Failed to free training labels on host", cudaFreeHost( trainingLabels ) );
	}

	CUDA_VERIFY(
		"Failed to allocate space for training vector squared norms on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceTrainingVectorNormsSquared ), ( m_clusters << m_logMaximumClusterSize ) * sizeof( float ) )
	);
	{	float* trainingVectorNormsSquared;
		CUDA_VERIFY(
			"Failed to allocate space for training vector squared norms on host",
			cudaMallocHost( &trainingVectorNormsSquared, ( m_clusters << m_logMaximumClusterSize ) * sizeof( float ) )
		);

		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();
			for ( unsigned int jj = 0; jj < size; ++jj )
				trainingVectorNormsSquared[ ( ii << m_logMaximumClusterSize ) + jj ] = m_trainingVectorNormsSquared[ m_clusterIndices[ ii ][ jj ] ];
			for ( unsigned int jj = size; jj < ( 1u << m_logMaximumClusterSize ); ++jj )
				trainingVectorNormsSquared[ ( ii << m_logMaximumClusterSize ) + jj ] = 0;
		}
		CUDA_VERIFY(
			"Failed to copy training vector squared norms to device",
			cudaMemcpy(
				m_deviceTrainingVectorNormsSquared,
				trainingVectorNormsSquared,
				( m_clusters << m_logMaximumClusterSize ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY( "Failed to free training vector squared norms on host", cudaFreeHost( trainingVectorNormsSquared ) );
	}

	CUDA_VERIFY(
		"Failed to allocate space for training vector kernel squared norms on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceTrainingVectorKernelNormsSquared ), ( m_clusters << m_logMaximumClusterSize ) * sizeof( float ) )
	);
	{	float* trainingVectorKernelNormsSquared;
		CUDA_VERIFY(
			"Failed to allocate space for training vector kernel squared norms on host",
			cudaMallocHost( &trainingVectorKernelNormsSquared, ( m_clusters << m_logMaximumClusterSize ) * sizeof( float ) )
		);

		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();
			for ( unsigned int jj = 0; jj < size; ++jj )
				trainingVectorKernelNormsSquared[ ( ii << m_logMaximumClusterSize ) + jj ] = m_trainingVectorKernelNormsSquared[ m_clusterIndices[ ii ][ jj ] ];
			for ( unsigned int jj = size; jj < ( 1u << m_logMaximumClusterSize ); ++jj )
				trainingVectorKernelNormsSquared[ ( ii << m_logMaximumClusterSize ) + jj ] = 0;
		}
		CUDA_VERIFY(
			"Failed to copy training vector kernel squared norms to device",
			cudaMemcpy(
				m_deviceTrainingVectorKernelNormsSquared,
				trainingVectorKernelNormsSquared,
				( m_clusters << m_logMaximumClusterSize ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY( "Failed to free training vector kernel squared norms on host", cudaFreeHost( trainingVectorKernelNormsSquared ) );
	}

	CUDA_VERIFY(
		"Failed to allocate space for training responses on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceTrainingResponses ), ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ) )
	);
	{	CUDA_FLOAT_DOUBLE* trainingResponses;
		CUDA_VERIFY(
			"Failed to allocate space for training responses on host",
			cudaMallocHost( &trainingResponses, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ) )
		);

		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();
			for ( unsigned int jj = 0; jj < size; ++jj )
				for ( unsigned int kk = 0; kk < m_classes; ++kk )
					trainingResponses[ ( ( ii * m_classes + kk ) << m_logMaximumClusterSize ) + jj ] = m_trainingResponses[ m_clusterIndices[ ii ][ jj ] * m_classes + kk ];
			for ( unsigned int jj = size; jj < ( 1u << m_logMaximumClusterSize ); ++jj )
				for ( unsigned int kk = 0; kk < m_classes; ++kk )
					trainingResponses[ ( ( ii * m_classes + kk ) << m_logMaximumClusterSize ) + jj ] = 0;
		}
		CUDA_VERIFY(
			"Failed to copy training responses to device",
			cudaMemcpy(
				m_deviceTrainingResponses,
				trainingResponses,
				( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY( "Failed to free training responses on host", cudaFreeHost( trainingResponses ) );
	}
	m_updatedResponses = true;

	CUDA_VERIFY(
		"Failed to allocate space for training alphas on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceTrainingAlphas ), ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( float ) )
	);
	{	if ( m_classes == 1 ) {

			for ( unsigned int ii = 0; ii < m_rows; ++ii )
				if ( ! ( m_trainingLabels[ ii ] > 0 ) )
					m_trainingAlphas[ ii ] = -m_trainingAlphas[ ii ];
		}

		float* trainingAlphas;
		CUDA_VERIFY(
			"Failed to allocate space for training alphas on host",
			cudaMallocHost( &trainingAlphas, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( float ) )
		);

		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();
			for ( unsigned int jj = 0; jj < size; ++jj )
				for ( unsigned int kk = 0; kk < m_classes; ++kk )
					trainingAlphas[ ( ( ii * m_classes + kk ) << m_logMaximumClusterSize ) + jj ] = m_trainingAlphas[ m_clusterIndices[ ii ][ jj ] * m_classes + kk ];
			for ( unsigned int jj = size; jj < ( 1u << m_logMaximumClusterSize ); ++jj )
				for ( unsigned int kk = 0; kk < m_classes; ++kk )
					trainingAlphas[ ( ( ii * m_classes + kk ) << m_logMaximumClusterSize ) + jj ] = 0;
		}
		CUDA_VERIFY(
			"Failed to copy training alphas to device",
			cudaMemcpy(
				m_deviceTrainingAlphas,
				trainingAlphas,
				( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY( "Failed to free training alphas on host", cudaFreeHost( trainingAlphas ) );

		if ( m_classes == 1 ) {

			for ( unsigned int ii = 0; ii < m_rows; ++ii )
				if ( ! ( m_trainingLabels[ ii ] > 0 ) )
					m_trainingAlphas[ ii ] = -m_trainingAlphas[ ii ];
		}
	}

	CUDA_VERIFY(
		"Failed to allocate space for cluster headers on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceClusterHeaders ), m_clusters * sizeof( CUDA::SparseKernelClusterHeader ) )
	);
	CUDA_VERIFY(
		"Failed to allocate space for cluster size sums on device",
		cudaMalloc( reinterpret_cast< void** >( &m_deviceClusterSizeSums ), ( m_clusters + 1 ) * sizeof( boost::uint32_t ) )
	);
	{	unsigned int totalClusterSize        = 0;
		unsigned int totalAlignedClusterSize = 0;
		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const dimension = m_clusterNonzeroIndices[ ii ].size();
			unsigned int const alignedDimension = ( ( dimension + 15 ) & ~15 );

			totalClusterSize += dimension;
			totalAlignedClusterSize += ( ( alignedDimension + 15 ) & ~15 );
		}

		CUDA_VERIFY(
			"Failed to allocate space for nonzero indices on device",
			cudaMalloc( reinterpret_cast< void** >( &m_deviceNonzeroIndices ), totalAlignedClusterSize * sizeof( boost::uint32_t ) )
		);
		CUDA_VERIFY(
			"Failed to allocate space for training vectors on device",
			cudaMalloc( reinterpret_cast< void** >( &m_deviceTrainingVectorsTranspose ), ( totalClusterSize << m_logMaximumClusterSize ) * sizeof( float ) )
		);
		boost::uint32_t* pDeviceNonzeroIndices           = m_deviceNonzeroIndices;
		float*    pDeviceTrainingVectorsTranspose = m_deviceTrainingVectorsTranspose;

		CUDA::SparseKernelClusterHeader* clusterHeaders;
		CUDA_VERIFY(
			"Failed to allocate space for cluster headers on host",
			cudaMallocHost( &clusterHeaders, m_clusters * sizeof( CUDA::SparseKernelClusterHeader ) )
		);

		boost::uint32_t* clusterSizeSums;
		CUDA_VERIFY(
			"Failed to allocate space for cluster size sums on host",
			cudaMallocHost( &clusterSizeSums, ( m_clusters + 1 ) * sizeof( boost::uint32_t ) )
		);

		boost::uint32_t* nonzeroIndices;
		CUDA_VERIFY(
			"Failed to allocate space for nonzero indices on host",
			cudaMallocHost( &nonzeroIndices, ( ( m_columns + 15 ) & ~15 ) * sizeof( boost::uint32_t ) )
		);

		float* trainingVectorsTranspose;
		CUDA_VERIFY(
			"Failed to allocate space for transposed training vectors on host",
			cudaMallocHost( &trainingVectorsTranspose, ( m_columns << m_logMaximumClusterSize ) * sizeof( float ) )
		);

		clusterSizeSums[ 0 ] = 0;
		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();

			unsigned int const dimension = m_clusterNonzeroIndices[ ii ].size();
			unsigned int const alignedDimension = ( ( dimension + 15 ) & ~15 );

			for ( unsigned int jj = 0; jj < dimension; ++jj )
				nonzeroIndices[ jj ] = m_clusterNonzeroIndices[ ii ][ jj ];
			for ( unsigned int jj = dimension; jj < alignedDimension; ++jj )
				nonzeroIndices[ jj ] = 0;
			CUDA_VERIFY(
				"Failed to copy nonzero indices to device",
				cudaMemcpy(
					pDeviceNonzeroIndices,
					nonzeroIndices,
					alignedDimension * sizeof( boost::uint32_t ),
					cudaMemcpyHostToDevice
				)
			);

			for ( unsigned int jj = 0; jj < size; ++jj ) {

				unsigned int const index = m_clusterIndices[ ii ][ jj ];

				unsigned int mm = 0;

				SparseVector::const_iterator kk    = m_trainingVectors[ index ].begin();
				SparseVector::const_iterator kkEnd = m_trainingVectors[ index ].end();

				std::vector< unsigned int >::const_iterator ll    = m_clusterNonzeroIndices[ ii ].begin();
				std::vector< unsigned int >::const_iterator llEnd = m_clusterNonzeroIndices[ ii ].end();

				while ( ( kk != kkEnd ) && ( ll != llEnd ) ) {

					BOOST_ASSERT( *ll <= kk->first );
					if ( *ll < kk->first ) {

						trainingVectorsTranspose[ ( mm << m_logMaximumClusterSize ) + jj ] = 0;
						++mm;
						++ll;
					}
					else {

						trainingVectorsTranspose[ ( mm << m_logMaximumClusterSize ) + jj ] = kk->second;
						++mm;
						++ll;
						++kk;
					}
				}
				for ( ; ll != llEnd; ++mm, ++ll )
					trainingVectorsTranspose[ ( mm << m_logMaximumClusterSize ) + jj ] = 0;
			}
			CUDA_VERIFY(
				"Failed to copy transposed training vectors to device",
				cudaMemcpy(
					pDeviceTrainingVectorsTranspose,
					trainingVectorsTranspose,
					( dimension << m_logMaximumClusterSize ) * sizeof( float ),
					cudaMemcpyHostToDevice
				)
			);

			clusterHeaders[ ii ].size = size;
			clusterHeaders[ ii ].nonzeros = m_clusterNonzeroIndices[ ii ].size();

			clusterHeaders[ ii ].responses = m_deviceTrainingResponses + ( ( ii * m_classes ) << m_logMaximumClusterSize );
			clusterHeaders[ ii ].labels = m_deviceTrainingLabels + ( ii << m_logMaximumClusterSize );
			clusterHeaders[ ii ].alphas = m_deviceTrainingAlphas + ( ( ii * m_classes ) << m_logMaximumClusterSize );

			clusterHeaders[ ii ].nonzeroIndices = pDeviceNonzeroIndices;
			clusterHeaders[ ii ].vectorsTranspose = pDeviceTrainingVectorsTranspose;
			clusterHeaders[ ii ].vectorNormsSquared = m_deviceTrainingVectorNormsSquared + ( ii << m_logMaximumClusterSize );
			clusterHeaders[ ii ].vectorKernelNormsSquared = m_deviceTrainingVectorKernelNormsSquared + ( ii << m_logMaximumClusterSize );

			clusterSizeSums[ ii + 1 ] = clusterSizeSums[ ii ] + size;

			pDeviceNonzeroIndices += alignedDimension;
			pDeviceTrainingVectorsTranspose += ( dimension << m_logMaximumClusterSize );;
		}

		CUDA_VERIFY(
			"Failed to copy cluster headers to device",
			cudaMemcpy(
				m_deviceClusterHeaders,
				clusterHeaders,
				m_clusters * sizeof( CUDA::SparseKernelClusterHeader ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy cluster size sums to device",
			cudaMemcpy(
				m_deviceClusterSizeSums,
				clusterSizeSums,
				( m_clusters + 1 ) * sizeof( boost::uint32_t ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY( "Failed to free cluster headers on host", cudaFreeHost( clusterHeaders ) );
		CUDA_VERIFY( "Failed to free cluster size sums on host", cudaFreeHost( clusterSizeSums ) );
		CUDA_VERIFY( "Failed to free nonzero indices on host", cudaFreeHost( nonzeroIndices ) );
		CUDA_VERIFY( "Failed to free transposed training vectors on host", cudaFreeHost( trainingVectorsTranspose ) );
	}

	m_workSize = std::max(
		( ( m_clusters + 15 ) >> 4 ) * std::max( sizeof( CUDA_FLOAT_DOUBLE ), sizeof( boost::uint32_t ) ),
		( ( ( m_clusters + 31 ) >> 5 ) << 9 ) * std::max( sizeof( float ), sizeof( boost::uint32_t ) )
	);
	BOOST_ASSERT( ( m_logMaximumClusterSize == 4 ) || ( m_logMaximumClusterSize == 8 ) );
	if ( m_logMaximumClusterSize == 4 )
		m_workSize = std::max( m_workSize, ( m_clusters << 8 ) * sizeof( CUDA_FLOAT_DOUBLE ) );
	else if ( m_logMaximumClusterSize == 8 )
		m_workSize = std::max( m_workSize, ( ( m_clusters * m_classes ) << 12 ) * sizeof( CUDA_FLOAT_DOUBLE ) );
	for ( unsigned int ii = 0; ii < ARRAYLENGTH( m_deviceWork ); ++ii ) {

		CUDA_VERIFY(
			"Failed to allocate space for work on device",
			cudaMalloc(
				&m_deviceWork[ ii ],
				m_workSize
			)
		);
	}
}


void SVM::UpdateResponses() {

	if ( m_initializedDevice && ( ! m_updatedResponses ) ) {

		CUDA_FLOAT_DOUBLE* trainingResponses;
		CUDA_VERIFY(
			"Failed to allocate space for training responses on host",
			cudaMallocHost( &trainingResponses, ( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ) )
		);

		CUDA_VERIFY(
			"Failed to copy training responses from device",
			cudaMemcpy(
				trainingResponses,
				m_deviceTrainingResponses,
				( ( m_clusters * m_classes ) << m_logMaximumClusterSize ) * sizeof( CUDA_FLOAT_DOUBLE ),
				cudaMemcpyDeviceToHost
			)
		);
		for ( unsigned int ii = 0; ii < m_clusters; ++ii ) {

			unsigned int const size = m_clusterIndices[ ii ].size();
			for ( unsigned int jj = 0; jj < size; ++jj )
				for ( unsigned int kk = 0; kk < m_classes; ++kk )
					m_trainingResponses[ m_clusterIndices[ ii ][ jj ] * m_classes + kk ] = trainingResponses[ ( ( ii * m_classes + kk ) << m_logMaximumClusterSize ) + jj ];
		}

		CUDA_VERIFY( "Failed to free training responses on host", cudaFreeHost( trainingResponses ) );

		m_updatedResponses = true;
	}
}


bool const SVM::IterateUnbiasedBinary() {

	BOOST_ASSERT( m_classes == 1 );

	bool progress = false;

	CUDA::SparseKernelFindLargestScore(
		m_foundKeys,
		m_foundValues,
		m_deviceWork[ 0 ],
		m_deviceWork[ 1 ],
		m_deviceWork[ 2 ],
		m_deviceWork[ 3 ],
		m_deviceClusterHeaders,
		m_logMaximumClusterSize,
		m_clusters,
		1,
		m_workSize,
		16,
		m_foundSize,
		m_regularization
	);
	std::copy( m_foundValues, m_foundValues + 16, m_foundIndices );

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_foundIndices[ ii ];
		unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		BOOST_ASSERT( unclusteredIndex < m_rows );

		for ( unsigned int jj = 0; jj < ii; ++jj )
			BOOST_ASSERT( m_batchIndices[ jj ] != batchIndex );

		m_batchIndices[ ii ] = batchIndex;

		unsigned int kk = 0;
		SparseVector::const_iterator jj    = m_trainingVectors[ unclusteredIndex ].begin();
		SparseVector::const_iterator jjEnd = m_trainingVectors[ unclusteredIndex ].end();
		for ( ; jj != jjEnd; ++jj ) {

			BOOST_ASSERT( jj->first < m_columns );
			for ( ; kk < jj->first; ++kk )
				m_batchVectorsTranspose[ kk * 16 + ii ] = 0;
			m_batchVectorsTranspose[ kk * 16 + ii ] = jj->second;
			++kk;
		}
		for ( ; kk < m_columns; ++kk )
			m_batchVectorsTranspose[ kk * 16 + ii ] = 0;

		m_batchVectorNormsSquared[ ii ] = m_trainingVectorNormsSquared[ unclusteredIndex ];
	}

	CUDA_VERIFY(
		"Failed to copy batch indices to device",
		cudaMemcpy(
			m_deviceBatchIndices,
			m_batchIndices,
			16 * sizeof( boost::uint32_t ),
			cudaMemcpyHostToDevice
		)
	);

#ifdef CUDA_USE_DOUBLE
	CUDA::DArrayRead(
		m_deviceBatchResponses,
		m_deviceTrainingResponses,
		m_deviceBatchIndices,
		16
	);
#else    // CUDA_USE_DOUBLE
	CUDA::FArrayRead(
		m_deviceBatchResponses,
		m_deviceTrainingResponses,
		m_deviceBatchIndices,
		16
	);
#endif    // CUDA_USE_DOUBLE

	CUDA_VERIFY(
		"Failed to copy batch responses from device",
		cudaMemcpy(
			m_batchResponses,
			m_deviceBatchResponses,
			16 * sizeof( CUDA_FLOAT_DOUBLE ),
			cudaMemcpyDeviceToHost
		)
	);

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const iiUnclusteredIndex = m_clusterIndices[ m_batchIndices[ ii ] >> m_logMaximumClusterSize ][ m_batchIndices[ ii ] & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		BOOST_ASSERT( iiUnclusteredIndex < m_rows );

		for ( unsigned int jj = 0; jj < ii; ++jj ) {

			unsigned int const jjUnclusteredIndex = m_clusterIndices[ m_batchIndices[ jj ] >> m_logMaximumClusterSize ][ m_batchIndices[ jj ] & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
			BOOST_ASSERT( jjUnclusteredIndex < m_rows );

			double accumulator = 0;

			SparseVector::const_iterator kk    = m_trainingVectors[ iiUnclusteredIndex ].begin();
			SparseVector::const_iterator kkEnd = m_trainingVectors[ iiUnclusteredIndex ].end();
			SparseVector::const_iterator ll    = m_trainingVectors[ jjUnclusteredIndex ].begin();
			SparseVector::const_iterator llEnd = m_trainingVectors[ jjUnclusteredIndex ].end();
			while ( ( kk != kkEnd ) && ( ll != llEnd ) ) {

				if ( kk->first < ll->first )
					++kk;
				else if ( kk->first > ll->first )
					++ll;
				else {

					accumulator += kk->second * ll->second;
					++kk;
					++ll;
				}
			}

			double value = std::numeric_limits< double >::quiet_NaN();
			switch( m_kernel ) {
				case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				default: throw std::runtime_error( "Unknown kernel" );
			}
			m_batchSubmatrix[ ( ii << 4 ) + jj ] = m_batchSubmatrix[ ( jj << 4 ) + ii ] = value;
		}
		double value = std::numeric_limits< double >::quiet_NaN();
		switch( m_kernel ) {
			case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			default: throw std::runtime_error( "Unknown kernel" );
		}
		m_batchSubmatrix[ ( ii << 4 ) + ii ] = value;
	}

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_batchIndices[ ii ];
		unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		m_batchAlphas[ ii ] = m_trainingAlphas[ unclusteredIndex ];
	}
	for ( unsigned int ii = 0; ii < 32; ++ii ) {

		double alpha = std::numeric_limits< double >::quiet_NaN();

		unsigned int bestIndex = 0;
		{	double bestScore = -std::numeric_limits< double >::infinity();
			for ( unsigned int jj = 0; jj < 16; ++jj ) {

				unsigned int const batchIndex = m_batchIndices[ jj ];
				unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
				float const sign = ( ( m_trainingLabels[ unclusteredIndex ] > 0 ) ? 1.0f : -1.0f );
				double const gradient = 1 - sign * m_batchResponses[ jj ];
				double const scale = m_batchSubmatrix[ ( jj << 4 ) + jj ];

				double newAlpha = m_batchAlphas[ jj ] + gradient / scale;
				if ( newAlpha > m_regularization )
					newAlpha = m_regularization;
				else if ( newAlpha < 0 )
					newAlpha = 0;

				double const delta = newAlpha - m_batchAlphas[ jj ];
				double const score = ( gradient - 0.5 * delta * scale ) * delta;

				if ( score > bestScore ) {

					bestIndex = jj;
					alpha = newAlpha;
					bestScore = score;
				}
			}
			if ( bestScore <= 0 )
				break;
		}

		unsigned int const batchIndex = m_batchIndices[ bestIndex ];
		unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		float const sign = ( ( m_trainingLabels[ unclusteredIndex ] > 0 ) ? 1.0f : -1.0f );

		if ( alpha != m_batchAlphas[ bestIndex ] ) {

			for ( unsigned int jj = 0; jj < 16; ++jj )
				m_batchResponses[ jj ] += ( alpha - m_batchAlphas[ bestIndex ] ) * sign * m_batchSubmatrix[ ( bestIndex << 4 ) + jj ];
			m_batchAlphas[ bestIndex ] = alpha;
		}
	}
	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_batchIndices[ ii ];
		unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		float const sign = ( ( m_trainingLabels[ unclusteredIndex ] > 0 ) ? 1.0f : -1.0f );

		if ( m_trainingAlphas[ unclusteredIndex ] != m_batchAlphas[ ii ] )
			progress = true;

		m_trainingAlphas[ unclusteredIndex ] = m_batchAlphas[ ii ];
		m_batchAlphas[ ii ] *= sign;
	}

	if ( progress ) {

		CUDA_VERIFY(
			"Failed to copy batch alphas to device",
			cudaMemcpy(
				m_deviceBatchAlphas,
				m_batchAlphas,
				16 * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch to device",
			cudaMemcpy(
				m_deviceBatchVectorsTranspose,
				m_batchVectorsTranspose,
				( m_columns << 4 ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch squared norms to device",
			cudaMemcpy(
				m_deviceBatchVectorNormsSquared,
				m_batchVectorNormsSquared,
				16 * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA::SparseUpdateKernel(
			m_deviceBatchVectorsTranspose,
			m_deviceBatchVectorNormsSquared,
			m_deviceBatchAlphas,
			m_deviceBatchIndices,
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			1,
			m_kernel,
			m_kernelParameter1,
			m_kernelParameter2,
			m_kernelParameter3
		);
	}

	return progress;
}


bool const SVM::IterateBiasedBinary() {

	BOOST_ASSERT( m_classes == 1 );

	bool progress = false;

	CUDA::SparseKernelFindLargestPositiveGradient(
		m_foundKeys,
		m_foundValues,
		m_deviceWork[ 0 ],
		m_deviceWork[ 1 ],
		m_deviceWork[ 2 ],
		m_deviceWork[ 3 ],
		m_deviceClusterHeaders,
		m_logMaximumClusterSize,
		m_clusters,
		m_workSize,
		16,
		m_foundSize,
		m_regularization
	);
	std::copy( m_foundValues, m_foundValues + 16, m_foundIndices );
	CUDA::SparseKernelFindLargestNegativeGradient(
		m_foundKeys,
		m_foundValues,
		m_deviceWork[ 0 ],
		m_deviceWork[ 1 ],
		m_deviceWork[ 2 ],
		m_deviceWork[ 3 ],
		m_deviceClusterHeaders,
		m_logMaximumClusterSize,
		m_clusters,
		m_workSize,
		16,
		m_foundSize,
		m_regularization
	);
	std::copy( m_foundValues, m_foundValues + 16, m_foundIndices + 16 );

	{	unsigned int ii = 0;
		for ( unsigned int jj = 0; jj < 32; ++jj ) {

			unsigned int const batchIndex = m_foundIndices[ ( ( jj & 1 ) ? 31 : 15 ) - ( jj >> 1 ) ];
			unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
			BOOST_ASSERT( unclusteredIndex < m_rows );

			bool duplicate = false;
			for ( unsigned int kk = 0; kk < ii; ++kk ) {

				if ( m_batchIndices[ kk ] == batchIndex ) {

					duplicate = true;
					break;
				}
			}

			if ( ! duplicate ) {

				BOOST_ASSERT( ii < 16 );

				m_batchIndices[ ii ] = batchIndex;

				unsigned int ll = 0;
				SparseVector::const_iterator kk    = m_trainingVectors[ unclusteredIndex ].begin();
				SparseVector::const_iterator kkEnd = m_trainingVectors[ unclusteredIndex ].end();
				for ( ; kk != kkEnd; ++kk ) {

					BOOST_ASSERT( kk->first < m_columns );
					for ( ; ll < kk->first; ++ll )
						m_batchVectorsTranspose[ ll * 16 + ii ] = 0;
					m_batchVectorsTranspose[ ll * 16 + ii ] = kk->second;
					++ll;
				}
				for ( ; ll < m_columns; ++ll )
					m_batchVectorsTranspose[ ll * 16 + ii ] = 0;

				m_batchVectorNormsSquared[ ii ] = m_trainingVectorNormsSquared[ unclusteredIndex ];

				if ( ++ii >= 16 )
					break;
			}
		}
		BOOST_ASSERT( ii == 16 );
	}

	CUDA_VERIFY(
		"Failed to copy batch indices to device",
		cudaMemcpy(
			m_deviceBatchIndices,
			m_batchIndices,
			16 * sizeof( boost::uint32_t ),
			cudaMemcpyHostToDevice
		)
	);

#ifdef CUDA_USE_DOUBLE
	CUDA::DArrayRead(
		m_deviceBatchResponses,
		m_deviceTrainingResponses,
		m_deviceBatchIndices,
		16
	);
#else    // CUDA_USE_DOUBLE
	CUDA::FArrayRead(
		m_deviceBatchResponses,
		m_deviceTrainingResponses,
		m_deviceBatchIndices,
		16
	);
#endif    // CUDA_USE_DOUBLE

	CUDA_VERIFY(
		"Failed to copy batch responses from device",
		cudaMemcpy(
			m_batchResponses,
			m_deviceBatchResponses,
			16 * sizeof( CUDA_FLOAT_DOUBLE ),
			cudaMemcpyDeviceToHost
		)
	);

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const iiUnclusteredIndex = m_clusterIndices[ m_batchIndices[ ii ] >> m_logMaximumClusterSize ][ m_batchIndices[ ii ] & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		BOOST_ASSERT( iiUnclusteredIndex < m_rows );

		for ( unsigned int jj = 0; jj < ii; ++jj ) {

			unsigned int const jjUnclusteredIndex = m_clusterIndices[ m_batchIndices[ jj ] >> m_logMaximumClusterSize ][ m_batchIndices[ jj ] & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
			BOOST_ASSERT( jjUnclusteredIndex < m_rows );

			double accumulator = 0;

			SparseVector::const_iterator kk    = m_trainingVectors[ iiUnclusteredIndex ].begin();
			SparseVector::const_iterator kkEnd = m_trainingVectors[ iiUnclusteredIndex ].end();
			SparseVector::const_iterator ll    = m_trainingVectors[ jjUnclusteredIndex ].begin();
			SparseVector::const_iterator llEnd = m_trainingVectors[ jjUnclusteredIndex ].end();
			while ( ( kk != kkEnd ) && ( ll != llEnd ) ) {

				if ( kk->first < ll->first )
					++kk;
				else if ( kk->first > ll->first )
					++ll;
				else {

					accumulator += kk->second * ll->second;
					++kk;
					++ll;
				}
			}

			double value = std::numeric_limits< double >::quiet_NaN();
			switch( m_kernel ) {
				case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				default: throw std::runtime_error( "Unknown kernel" );
			}
			m_batchSubmatrix[ ( ii << 4 ) + jj ] = m_batchSubmatrix[ ( jj << 4 ) + ii ] = value;
		}
		double value = std::numeric_limits< double >::quiet_NaN();
		switch( m_kernel ) {
			case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			default: throw std::runtime_error( "Unknown kernel" );
		}
		m_batchSubmatrix[ ( ii << 4 ) + ii ] = value;
	}

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_batchIndices[ ii ];
		unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		m_batchAlphas[ ii ] = m_trainingAlphas[ unclusteredIndex ];
	}
	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int bestIndex1 = 0;
		{	double bestScore = -std::numeric_limits< double >::infinity();
			for ( unsigned int jj = 0; jj < 16; ++jj ) {

				unsigned int const batchIndex = m_batchIndices[ jj ];
				unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
				float const sign = ( ( m_trainingLabels[ unclusteredIndex ] > 0 ) ? 1.0f : -1.0f );
				double const gradient = 1 - sign * m_batchResponses[ jj ];

				float score = std::abs( gradient );
				if ( ( gradient > 0 ) && ( ! ( m_batchAlphas[ jj ] < m_regularization ) ) )
					score = -score;
				else if ( ( gradient < 0 ) && ( ! ( m_batchAlphas[ jj ] > 0 ) ) )
					score = -score;

				if ( score > bestScore ) {

					bestIndex1 = jj;
					bestScore = score;
				}
			}
		}
		unsigned int const batchIndex1 = m_batchIndices[ bestIndex1 ];
		unsigned int const unclusteredIndex1 = m_clusterIndices[ batchIndex1 >> m_logMaximumClusterSize ][ batchIndex1 & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		float const sign1 = ( ( m_trainingLabels[ unclusteredIndex1 ] > 0 ) ? 1.0f : -1.0f );

		double alpha1 = std::numeric_limits< double >::quiet_NaN();
		double alpha2 = std::numeric_limits< double >::quiet_NaN();

		unsigned int bestIndex2 = 0;
		{	double bestScore = -std::numeric_limits< double >::infinity();
			for ( unsigned int jj = 0; jj < 16; ++jj ) {

				if ( jj != bestIndex1 ) {

					unsigned int const batchIndex = m_batchIndices[ jj ];
					unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
					float const sign = ( ( m_trainingLabels[ unclusteredIndex ] > 0 ) ? 1.0f : -1.0f );

					double const k11 = m_batchSubmatrix[ ( bestIndex1 << 4 ) + bestIndex1 ];
					double const k22 = m_batchSubmatrix[ ( jj << 4 ) + jj ];
					double const k12 = m_batchSubmatrix[ ( bestIndex1 << 4 ) + jj ];
					double delta = (
						( ( sign1 - sign ) - ( m_batchResponses[ bestIndex1 ] - m_batchResponses[ jj ] ) ) /
						std::max( k11 + k22 - 2 * k12, static_cast< double >( std::numeric_limits< float >::epsilon() ) )
					);

					double newAlpha1 = m_batchAlphas[ bestIndex1 ] + delta * sign1;
					if ( newAlpha1 < 0 )
						newAlpha1 = 0;
					else if ( newAlpha1 > m_regularization )
						newAlpha1 = m_regularization;
					double const positiveDelta = ( newAlpha1 - m_batchAlphas[ bestIndex1 ] ) * sign1;

					double newAlpha2 = m_batchAlphas[ jj ] - delta * sign;
					if ( newAlpha2 < 0 )
						newAlpha2 = 0;
					else if ( newAlpha2 > m_regularization )
						newAlpha2 = m_regularization;
					double const negativeDelta = ( m_batchAlphas[ jj ] - newAlpha2 ) * sign;

					if ( std::abs( positiveDelta ) < std::abs( negativeDelta ) ) {

						delta = positiveDelta;
						newAlpha2 = m_batchAlphas[ jj ] - delta * sign;
						BOOST_ASSERT( ( newAlpha2 >= 0 ) && ( newAlpha2 <= m_regularization ) );
					}
					else if ( std::abs( negativeDelta ) < std::abs( positiveDelta ) ) {

						delta = negativeDelta;
						newAlpha1 = m_batchAlphas[ bestIndex1 ] + delta * sign1;
						BOOST_ASSERT( ( newAlpha2 >= 0 ) && ( newAlpha2 <= m_regularization ) );
					}
					else {

						BOOST_ASSERT( positiveDelta == negativeDelta );
						delta = positiveDelta;
					}

					double const score = (
						( ( sign1 - sign ) - ( m_batchResponses[ bestIndex1 ] - m_batchResponses[ jj ] ) ) * delta -
						0.5 * Square( delta ) * ( k11 + k22 - 2 * k12 )
					);

					if ( score > bestScore ) {

						bestIndex2 = jj;
						alpha1 = newAlpha1;
						alpha2 = newAlpha2;
						bestScore = score;
					}
				}
			}
			if ( bestScore <= 0 )
				break;
		}

		unsigned int const batchIndex2 = m_batchIndices[ bestIndex2 ];
		unsigned int const unclusteredIndex2 = m_clusterIndices[ batchIndex2 >> m_logMaximumClusterSize ][ batchIndex2 & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		float const sign2 = ( ( m_trainingLabels[ unclusteredIndex2 ] > 0 ) ? 1.0f : -1.0f );

		if ( alpha1 != m_batchAlphas[ bestIndex1 ] ) {

			for ( unsigned int jj = 0; jj < 16; ++jj )
				m_batchResponses[ jj ] += ( alpha1 - m_batchAlphas[ bestIndex1 ] ) * sign1 * m_batchSubmatrix[ ( bestIndex1 << 4 ) + jj ];
			m_batchAlphas[ bestIndex1 ] = alpha1;
		}
		if ( alpha2 != m_batchAlphas[ bestIndex2 ] ) {

			for ( unsigned int jj = 0; jj < 16; ++jj )
				m_batchResponses[ jj ] += ( alpha2 - m_batchAlphas[ bestIndex2 ] ) * sign2 * m_batchSubmatrix[ ( bestIndex2 << 4 ) + jj ];
			m_batchAlphas[ bestIndex2 ] = alpha2;
		}
	}
	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_batchIndices[ ii ];
		unsigned int const unclusteredIndex = m_clusterIndices[ batchIndex >> m_logMaximumClusterSize ][ batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		float const sign = ( ( m_trainingLabels[ unclusteredIndex ] > 0 ) ? 1.0f : -1.0f );

		if ( m_trainingAlphas[ unclusteredIndex ] != m_batchAlphas[ ii ] )
			progress = true;

		m_trainingAlphas[ unclusteredIndex ] = m_batchAlphas[ ii ];
		m_batchAlphas[ ii ] *= sign;
	}

	if ( progress ) {

		CUDA_VERIFY(
			"Failed to copy batch alphas to device",
			cudaMemcpy(
				m_deviceBatchAlphas,
				m_batchAlphas,
				16 * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch to device",
			cudaMemcpy(
				m_deviceBatchVectorsTranspose,
				m_batchVectorsTranspose,
				( m_columns << 4 ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch squared norms to device",
			cudaMemcpy(
				m_deviceBatchVectorNormsSquared,
				m_batchVectorNormsSquared,
				16 * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA::SparseUpdateKernel(
			m_deviceBatchVectorsTranspose,
			m_deviceBatchVectorNormsSquared,
			m_deviceBatchAlphas,
			m_deviceBatchIndices,
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			1,
			m_kernel,
			m_kernelParameter1,
			m_kernelParameter2,
			m_kernelParameter3
		);
	}

	return progress;
}


bool const SVM::IterateUnbiasedMulticlass() {

	BOOST_ASSERT( m_classes > 1 );

	bool progress = false;

	CUDA::SparseKernelFindLargestScore(
		m_foundKeys,
		m_foundValues,
		m_deviceWork[ 0 ],
		m_deviceWork[ 1 ],
		m_deviceWork[ 2 ],
		m_deviceWork[ 3 ],
		m_deviceClusterHeaders,
		m_logMaximumClusterSize,
		m_clusters,
		m_classes,
		m_workSize,
		16,
		m_foundSize,
		m_regularization
	);
	std::copy( m_foundValues, m_foundValues + 16, m_foundIndices );

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_foundIndices[ ii ];

		unsigned int const cluster = ( batchIndex >> m_logMaximumClusterSize );
		unsigned int const index = ( batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) );
		BOOST_ASSERT( cluster < m_clusters );
		BOOST_ASSERT( index < m_clusterIndices[ cluster ].size() );

		unsigned int const unclusteredIndex = m_clusterIndices[ cluster ][ index ];
		BOOST_ASSERT( unclusteredIndex < m_rows );

		for ( unsigned int jj = 0; jj < ii; ++jj )
			BOOST_ASSERT( m_foundIndices[ jj ] != batchIndex );

		for ( unsigned int jj = 0; jj < m_classes; ++jj )
			m_batchIndices[ jj * 16 + ii ] = ( ( cluster * m_classes + jj ) << m_logMaximumClusterSize ) + index;

		unsigned int kk = 0;
		SparseVector::const_iterator jj    = m_trainingVectors[ unclusteredIndex ].begin();
		SparseVector::const_iterator jjEnd = m_trainingVectors[ unclusteredIndex ].end();
		for ( ; jj != jjEnd; ++jj ) {

			BOOST_ASSERT( jj->first < m_columns );
			for ( ; kk < jj->first; ++kk )
				m_batchVectorsTranspose[ kk * 16 + ii ] = 0;
			m_batchVectorsTranspose[ kk * 16 + ii ] = jj->second;
			++kk;
		}
		for ( ; kk < m_columns; ++kk )
			m_batchVectorsTranspose[ kk * 16 + ii ] = 0;

		m_batchVectorNormsSquared[ ii ] = m_trainingVectorNormsSquared[ unclusteredIndex ];
	}

	CUDA_VERIFY(
		"Failed to copy batch indices to device",
		cudaMemcpy(
			m_deviceBatchIndices,
			m_batchIndices,
			16 * m_classes * sizeof( boost::uint32_t ),
			cudaMemcpyHostToDevice
		)
	);

#ifdef CUDA_USE_DOUBLE
	CUDA::DArrayRead(
		m_deviceBatchResponses,
		m_deviceTrainingResponses,
		m_deviceBatchIndices,
		16 * m_classes
	);
#else    // CUDA_USE_DOUBLE
	CUDA::FArrayRead(
		m_deviceBatchResponses,
		m_deviceTrainingResponses,
		m_deviceBatchIndices,
		16 * m_classes
	);
#endif    // CUDA_USE_DOUBLE

	CUDA_VERIFY(
		"Failed to copy batch responses from device",
		cudaMemcpy(
			m_batchResponses,
			m_deviceBatchResponses,
			16 * m_classes * sizeof( CUDA_FLOAT_DOUBLE ),
			cudaMemcpyDeviceToHost
		)
	);

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const iiUnclusteredIndex = m_clusterIndices[ m_foundIndices[ ii ] >> m_logMaximumClusterSize ][ m_foundIndices[ ii ] & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
		BOOST_ASSERT( iiUnclusteredIndex < m_rows );

		for ( unsigned int jj = 0; jj < ii; ++jj ) {

			unsigned int const jjUnclusteredIndex = m_clusterIndices[ m_foundIndices[ jj ] >> m_logMaximumClusterSize ][ m_foundIndices[ jj ] & ( ( 1u << m_logMaximumClusterSize ) - 1 ) ];
			BOOST_ASSERT( jjUnclusteredIndex < m_rows );

			double accumulator = 0;

			SparseVector::const_iterator kk    = m_trainingVectors[ iiUnclusteredIndex ].begin();
			SparseVector::const_iterator kkEnd = m_trainingVectors[ iiUnclusteredIndex ].end();
			SparseVector::const_iterator ll    = m_trainingVectors[ jjUnclusteredIndex ].begin();
			SparseVector::const_iterator llEnd = m_trainingVectors[ jjUnclusteredIndex ].end();
			while ( ( kk != kkEnd ) && ( ll != llEnd ) ) {

				if ( kk->first < ll->first )
					++kk;
				else if ( kk->first > ll->first )
					++ll;
				else {

					accumulator += kk->second * ll->second;
					++kk;
					++ll;
				}
			}

			double value = std::numeric_limits< double >::quiet_NaN();
			switch( m_kernel ) {
				case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( accumulator, m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ jj ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
				default: throw std::runtime_error( "Unknown kernel" );
			}
			m_batchSubmatrix[ ( ii << 4 ) + jj ] = m_batchSubmatrix[ ( jj << 4 ) + ii ] = value;
		}
		double value = std::numeric_limits< double >::quiet_NaN();
		switch( m_kernel ) {
			case GTSVM_KERNEL_GAUSSIAN:   { value = Kernel< GTSVM_KERNEL_GAUSSIAN   >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_POLYNOMIAL: { value = Kernel< GTSVM_KERNEL_POLYNOMIAL >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			case GTSVM_KERNEL_SIGMOID:    { value = Kernel< GTSVM_KERNEL_SIGMOID    >::Calculate( m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_batchVectorNormsSquared[ ii ], m_kernelParameter1, m_kernelParameter2, m_kernelParameter3 ); break; }
			default: throw std::runtime_error( "Unknown kernel" );
		}
		m_batchSubmatrix[ ( ii << 4 ) + ii ] = value;
	}

	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_foundIndices[ ii ];

		unsigned int const cluster = ( batchIndex >> m_logMaximumClusterSize );
		unsigned int const index = ( batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) );
		BOOST_ASSERT( cluster < m_clusters );
		BOOST_ASSERT( index < m_clusterIndices[ cluster ].size() );

		unsigned int const unclusteredIndex = m_clusterIndices[ cluster ][ index ];
		BOOST_ASSERT( unclusteredIndex < m_rows );

		for ( unsigned int jj = 0; jj < m_classes; ++jj )
			m_batchAlphas[ jj * 16 + ii ] = m_trainingAlphas[ unclusteredIndex * m_classes + jj ];
	}
	for ( unsigned int ii = 0; ii < 16 * m_classes; ++ii ) {

		unsigned int bestIndex = 0;
		double bestScore = -std::numeric_limits< double >::infinity();
		unsigned int bestMaximumIndex = 0;
		unsigned int bestMinimumIndex = 0;
		float bestMaximumAlpha = 0;
		float bestMinimumAlpha = 0;
		for ( unsigned int jj = 0; jj < 16; ++jj ) {

			unsigned int const batchIndex = m_foundIndices[ jj ];

			unsigned int const cluster = ( batchIndex >> m_logMaximumClusterSize );
			unsigned int const index = ( batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) );
			BOOST_ASSERT( cluster < m_clusters );
			BOOST_ASSERT( index < m_clusterIndices[ cluster ].size() );

			unsigned int const unclusteredIndex = m_clusterIndices[ cluster ][ index ];
			BOOST_ASSERT( unclusteredIndex < m_rows );

			unsigned int const label = m_trainingLabels[ unclusteredIndex ];

			double minimumGradient =  std::numeric_limits< double >::infinity();
			unsigned int minimumIndex = 0;

			for ( unsigned int kk = 0; kk < m_classes; ++kk ) {

				double gradient = -m_batchResponses[ kk * 16 + jj ];
				if ( kk == label )
					gradient += 1;

				if ( gradient < minimumGradient ) {

					minimumGradient = gradient;
					minimumIndex = kk;
				}
			}

			for ( unsigned int kk = 0; kk < m_classes; ++kk ) {

				double gradient = -m_batchResponses[ kk * 16 + jj ];
				float bound = 0;
				if ( kk == label ) {

					gradient += 1;
					bound = m_regularization;
				}

				if ( m_batchAlphas[ kk * 16 + jj ] < bound ) {

					double delta = 0.5 * ( gradient - minimumGradient ) / m_batchSubmatrix[ ( jj << 4 ) + jj ];
					if ( delta > 0 ) {

						BOOST_ASSERT( kk != minimumIndex );

						float maximumAlpha = m_batchAlphas[ kk * 16 + jj ] + delta;
						if ( maximumAlpha >= bound ) {

							maximumAlpha = bound;
							delta = bound - m_batchAlphas[ kk * 16 + jj ];
						}
						float minimumAlpha = m_batchAlphas[ minimumIndex * 16 + jj ] - delta;

						double const score = ( ( gradient - minimumGradient ) - delta * m_batchSubmatrix[ ( jj << 4 ) + jj ] ) * delta;
						if ( score > bestScore ) {

							bestIndex = jj;
							bestScore = score;
							bestMaximumIndex = kk;
							bestMinimumIndex = minimumIndex;
							bestMaximumAlpha = maximumAlpha;
							bestMinimumAlpha = minimumAlpha;
						}
					}
				}
			}
		}
		if ( bestScore <= 0 )
			break;

		for ( unsigned int jj = 0; jj < 16; ++jj ) {

			m_batchResponses[ bestMaximumIndex * 16 + jj ] += ( bestMaximumAlpha - m_batchAlphas[ bestMaximumIndex * 16 + bestIndex ] ) * m_batchSubmatrix[ ( bestIndex << 4 ) + jj ];
			m_batchResponses[ bestMinimumIndex * 16 + jj ] += ( bestMinimumAlpha - m_batchAlphas[ bestMinimumIndex * 16 + bestIndex ] ) * m_batchSubmatrix[ ( bestIndex << 4 ) + jj ];
		}
		m_batchAlphas[ bestMaximumIndex * 16 + bestIndex ] = bestMaximumAlpha;
		m_batchAlphas[ bestMinimumIndex * 16 + bestIndex ] = bestMinimumAlpha;
	}
	for ( unsigned int ii = 0; ii < 16; ++ii ) {

		unsigned int const batchIndex = m_foundIndices[ ii ];

		unsigned int const cluster = ( batchIndex >> m_logMaximumClusterSize );
		unsigned int const index = ( batchIndex & ( ( 1u << m_logMaximumClusterSize ) - 1 ) );
		BOOST_ASSERT( cluster < m_clusters );
		BOOST_ASSERT( index < m_clusterIndices[ cluster ].size() );

		unsigned int const unclusteredIndex = m_clusterIndices[ cluster ][ index ];
		BOOST_ASSERT( unclusteredIndex < m_rows );

		for ( unsigned int jj = 0; jj < m_classes; ++jj ) {

			if ( m_trainingAlphas[ unclusteredIndex * m_classes + jj ] != m_batchAlphas[ jj * 16 + ii ] )
				progress = true;

			m_trainingAlphas[ unclusteredIndex * m_classes + jj ] = m_batchAlphas[ jj * 16 + ii ];
		}
	}

	if ( progress ) {

		CUDA_VERIFY(
			"Failed to copy batch alphas to device",
			cudaMemcpy(
				m_deviceBatchAlphas,
				m_batchAlphas,
				16 * m_classes * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch to device",
			cudaMemcpy(
				m_deviceBatchVectorsTranspose,
				m_batchVectorsTranspose,
				( m_columns << 4 ) * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch squared norms to device",
			cudaMemcpy(
				m_deviceBatchVectorNormsSquared,
				m_batchVectorNormsSquared,
				16 * sizeof( float ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA_VERIFY(
			"Failed to copy batch indices to device",
			cudaMemcpy(
				m_deviceBatchIndices,
				m_foundIndices,
				16 * sizeof( boost::uint32_t ),
				cudaMemcpyHostToDevice
			)
		);

		CUDA::SparseUpdateKernel(
			m_deviceBatchVectorsTranspose,
			m_deviceBatchVectorNormsSquared,
			m_deviceBatchAlphas,
			m_deviceBatchIndices,
			m_deviceClusterHeaders,
			m_logMaximumClusterSize,
			m_clusters,
			m_classes,
			m_kernel,
			m_kernelParameter1,
			m_kernelParameter2,
			m_kernelParameter3
		);
	}

	return progress;
}




}    // namespace GTSVM
