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
	\file cuda.hpp
	\brief Includes all CUDA headers
*/




#ifndef __CUDA_HPP__
#define __CUDA_HPP__

#ifdef __cplusplus




/**
	\namespace CUDA
	\brief CUDA namespace
*/




#include "cuda_sparse_kernel.hpp"
#include "cuda_reduce.hpp"
#include "cuda_find_largest.hpp"
#include "cuda_partial_sum.hpp"
#include "cuda_array.hpp"

#include "cuda_exception.hpp"
#include "cuda_helpers.hpp"




#endif    /* __cplusplus */

#endif    /* __CUDA_HPP__ */
