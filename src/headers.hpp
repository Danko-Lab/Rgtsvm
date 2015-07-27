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
	\file headers.hpp
	\brief Precompiled header
*/




#ifndef __HEADERS_HPP__
#define __HEADERS_HPP__




/**
	\namespace _Private
	\brief _Private namespace
*/




#include <cuda_runtime.h>


#include "gtsvm.h"
#include "svm.hpp"
#include "cuda.hpp"
#include "helpers.hpp"


#include <boost/math/special_functions.hpp>
#include <boost/type_traits.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/cstdint.hpp>
#include <boost/version.hpp>


#include <string>
#include <map>
#include <set>
#include <vector>
#include <queue>

#include <sstream>
#include <iostream>
#include <fstream>

#include <stdexcept>
#include <algorithm>
#include <limits>
#include <memory>

#include <cstdlib>
#include <cstddef>
#include <cmath>


#include <math.h>




#endif    /* __HEADERS_HPP__ */
