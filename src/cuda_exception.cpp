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
	\file cuda_exception.cpp
	\brief implementation of CUDA::Exception class
*/




#include "headers.hpp"




namespace GTSVM {




namespace CUDA {




//============================================================================
//    Exception methods
//============================================================================


Exception::~Exception() throw() {
}


char const* Exception::what() const throw() {

	return m_what.c_str();
}




}    // namespace CUDA




}    // namespace GTSVM
