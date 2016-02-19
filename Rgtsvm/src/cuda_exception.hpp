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
	\file cuda_exception.hpp
	\brief definition of CUDA::Exception class
*/




#ifndef __CUDA_ERROR_HPP__
#define __CUDA_ERROR_HPP__

#ifdef __cplusplus




#include <sstream>
#include <stdexcept>




namespace GTSVM {




namespace CUDA {




//============================================================================
//    Exception class
//============================================================================


struct Exception : public std::exception {

	inline Exception(
		std::string const& file,
		unsigned int const line,
		std::string const& what,
		cudaError_t const& code
	);

	virtual ~Exception() throw();

	virtual char const* what() const throw();


private:

	std::string m_what;
};




//============================================================================
//    Exception inline methods
//============================================================================


Exception::Exception(
	std::string const& file,
	unsigned int const line,
	std::string const& what,
	cudaError_t const& code
)
{
	std::stringstream stream;
	stream << file << ':' << line << ": " << what << " (" << cudaGetErrorString( code ) << ')';
	m_what = stream.str();
}




}    // namespace CUDA




}    // namespace GTSVM




//============================================================================
//    CUDA_VERIFY macro
//============================================================================


#define CUDA_VERIFY( what, expression ) {  \
	cudaError_t const error = ( expression );  \
	if ( error != cudaSuccess )  \
		throw ::GTSVM::CUDA::Exception( __FILE__, __LINE__, what, error );  \
}




//============================================================================
//    CUDA_CHECK macro
//============================================================================


#define CUDA_CHECK( what )  \
	CUDA_VERIFY( what, cudaGetLastError() )




#endif    /* __cplusplus */

#endif    /* __CUDA_ERROR_HPP__ */
