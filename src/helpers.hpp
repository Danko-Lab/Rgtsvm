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
	\file helpers.hpp
	\brief Helper macros, functions and classes
*/




#ifndef __HELPERS_HPP__
#define __HELPERS_HPP__




#include <boost/static_assert.hpp>
#include <boost/cstdint.hpp>

#include <cmath>




namespace GTSVM {




//============================================================================
//    ARRAYLENGTH macro
//============================================================================


#define ARRAYLENGTH( array )  \
	( sizeof( array ) / sizeof( array[ 0 ] ) )




//============================================================================
//    LIKELY and UNLIKELY macros
//============================================================================


#if defined( __GNUC__ ) && ( __GNUC__ >= 3 )

#define LIKELY( boolean ) __builtin_expect( ( boolean ), 1 )
#define UNLIKELY( boolean ) __builtin_expect( ( boolean ), 0 )

#else    /* defined( __GNUC__ ) && ( __GNUC__ >= 3 ) */

#define LIKELY( boolean ) ( boolean )
#define UNLIKELY( boolean ) ( boolean )

#endif    /* defined( __GNUC__ ) && ( __GNUC__ >= 3 ) */




#ifdef __cplusplus




//============================================================================
//    Power helper template
//============================================================================


template< unsigned int t_Number, unsigned int t_Power >
struct Power {

	enum { RESULT = t_Number * Power< t_Number, t_Power - 1 >::RESULT };
};


template< unsigned int t_Number >
struct Power< t_Number, 0 > {

	enum { RESULT = 1 };
};




//============================================================================
//    Signum helper functions
//============================================================================


template< typename t_Type >
inline t_Type Signum( t_Type const& value ) {

	t_Type result = 0;
	if ( value < 0 )
		result = -1;
	else if ( value > 0 )
		result = 1;
	return result;
}


template< typename t_Type >
inline t_Type Signum( t_Type const& value, t_Type const& scale ) {

	t_Type result = 0;
	if ( value < 0 )
		result = -scale;
	else if ( value > 0 )
		result = scale;
	return result;
}




//============================================================================
//    Square helper function
//============================================================================


template< typename t_Type >
inline t_Type Square( t_Type const& value ) {

	return( value * value );
}




//============================================================================
//    Cube helper function
//============================================================================


template< typename t_Type >
inline t_Type Cube( t_Type const& value ) {

	return( value * value * value );
}




//============================================================================
//    CountBits helper functions
//============================================================================


inline unsigned char CountBits( boost::uint8_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 1 );
	number = ( number & 0x55 ) + ( ( number >> 1 ) & 0x55 );
	number = ( number & 0x33 ) + ( ( number >> 2 ) & 0x33 );
	number = ( number & 0x0f ) +   ( number >> 4 );
	return number;
}


inline unsigned short CountBits( boost::uint16_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 2 );
	number = ( number & 0x5555 ) + ( ( number >> 1 ) & 0x5555 );
	number = ( number & 0x3333 ) + ( ( number >> 2 ) & 0x3333 );
	number = ( number & 0x0f0f ) + ( ( number >> 4 ) & 0x0f0f );
	number = ( number & 0x00ff ) +   ( number >> 8 );
	return number;
}


inline unsigned int CountBits( boost::uint32_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 4 );
	number = ( number & 0x55555555 ) + ( ( number >>  1 ) & 0x55555555 );
	number = ( number & 0x33333333 ) + ( ( number >>  2 ) & 0x33333333 );
	number = ( number & 0x0f0f0f0f ) + ( ( number >>  4 ) & 0x0f0f0f0f );
	number = ( number & 0x00ff00ff ) + ( ( number >>  8 ) & 0x00ff00ff );
	number = ( number & 0x0000ffff ) +   ( number >> 16 );
	return number;
}


inline unsigned long long CountBits( boost::uint64_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 8 );
	number = ( number & 0x5555555555555555ull ) + ( ( number >>  1 ) & 0x5555555555555555ull );
	number = ( number & 0x3333333333333333ull ) + ( ( number >>  2 ) & 0x3333333333333333ull );
	number = ( number & 0x0f0f0f0f0f0f0f0full ) + ( ( number >>  4 ) & 0x0f0f0f0f0f0f0f0full );
	number = ( number & 0x00ff00ff00ff00ffull ) + ( ( number >>  8 ) & 0x00ff00ff00ff00ffull );
	number = ( number & 0x0000ffff0000ffffull ) + ( ( number >> 16 ) & 0x0000ffff0000ffffull );
	number = ( number & 0x00000000ffffffffull ) +   ( number >> 32 );
	return number;
}




//============================================================================
//    HighBit helper functions
//============================================================================


inline unsigned int HighBit( boost::uint8_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 1 );

	unsigned int bit = 0;

	if ( number & 0xf0 ) {

		bit += 4;
		number >>= 4;
	}

	if ( number & 0x0c ) {

		bit += 2;
		number >>= 2;
	}

	if ( number & 0x02 )
		++bit;

	return bit;
}


inline unsigned int HighBit( boost::uint16_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 2 );

	unsigned int bit = 0;

	if ( number & 0xff00 ) {

		bit += 8;
		number >>= 8;
	}

	if ( number & 0x00f0 ) {

		bit += 4;
		number >>= 4;
	}

	if ( number & 0x000c ) {

		bit += 2;
		number >>= 2;
	}

	if ( number & 0x0002 )
		++bit;

	return bit;
}


inline unsigned int HighBit( boost::uint32_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 4 );

	unsigned int bit = 0;

	if ( number & 0xffff0000 ) {

		bit += 16;
		number >>= 16;
	}

	if ( number & 0x0000ff00 ) {

		bit += 8;
		number >>= 8;
	}

	if ( number & 0x000000f0 ) {

		bit += 4;
		number >>= 4;
	}

	if ( number & 0x0000000c ) {

		bit += 2;
		number >>= 2;
	}

	if ( number & 0x00000002 )
		++bit;

	return bit;
}


inline unsigned int HighBit( boost::uint64_t number ) {

	BOOST_STATIC_ASSERT( sizeof( number ) == 8 );

	unsigned int bit = 0;

	if ( number & 0xffffffff00000000ull ) {

		bit += 32;
		number >>= 32;
	}

	if ( number & 0x00000000ffff0000ull ) {

		bit += 16;
		number >>= 16;
	}

	if ( number & 0x000000000000ff00ull ) {

		bit += 8;
		number >>= 8;
	}

	if ( number & 0x00000000000000f0ull ) {

		bit += 4;
		number >>= 4;
	}

	if ( number & 0x000000000000000cull ) {

		bit += 2;
		number >>= 2;
	}

	if ( number & 0x0000000000000002ull )
		++bit;

	return bit;
}




}    // namespace GTSVM




#endif    /* __cplusplus */




#endif    /* __HELPERS_HPP__ */
