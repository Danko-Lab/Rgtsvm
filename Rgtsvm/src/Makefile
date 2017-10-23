##	Copyright (C) 2017  Zhong Wang
##
##	This program is free software: you can redistribute it and/or modify
##	it under the terms of the GNU General Public License as published by
##	the Free Software Foundation, either version 3 of the License, or
##	(at your option) any later version.
##
##	This program is distributed in the hope that it will be useful,
##	but WITHOUT ANY WARRANTY; without even the implied warranty of
##	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##	GNU General Public License for more details.
##
##	You should have received a copy of the GNU General Public License
##	along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Compilers
EXT := cu

CXX := g++
NVCC := /usr/local/cuda-8.0/bin/nvcc
CXXFLAGS := -O2 -funroll-loops -fomit-frame-pointer -ffast-math -I/usr/local/cuda-8.0/include -I/usr/include -I/programs/R-3.3.2/include -Wall -fPIC -D_GNU_SOURCE -m64
NVCCFLAGS := -arch=sm_20 -O2 -I/usr/local/cuda-8.0/include -I/usr/include -DBOOST_NOINLINE='__attribute__((noinline))' -Xcompiler "-O2 -fpic -I/programs/R-3.3.2/include"
CFLAGS := -std=gnu99 -O2 -funroll-loops -fomit-frame-pointer -ffast-math -I/usr/local/cuda-8.0/include -I/usr/include -I/programs/R-3.3.2/include -Wall -fPIC -D_GNU_SOURCE -m64
NCFLAGS := -arch=sm_20 -O2 -I/usr/local/cuda-8.0/include -I/usr/include -DBOOST_NOINLINE='__attribute__((noinline))' -Xcompiler "-O2 -fpic -I/programs/R-3.3.2/include"
LDFLAGS := -L/usr/local/cuda-8.0/lib64
LIBS := -lcuda

# Program-specifics
TARGET := Rgtsvm.so
OBJS := Rgtsvm.o

HEADERS := \
	svm.hpp \
	cuda.hpp \
	cuda_sparse_kernel.hpp \
	cuda_reduce.hpp \
	cuda_find_largest.hpp \
	cuda_partial_sum.hpp \
	cuda_array.hpp \
	cuda_exception.hpp \
	cuda_helpers.hpp \
	helpers.hpp

SOURCES := \
	Rgtsvm.cpp \
	svmlight.cpp \
	svm.cpp \
	cuda_sparse_kernel.cu \
	cuda_reduce.cu \
	cuda_find_largest.cu \
	cuda_partial_sum.cu \
	cuda_array.cu \
	cuda_exception.cpp

PRECOMPILED_HEADER_SOURCE := \
	headers.hpp


#====  derived variables  =====================================================

PRECOMPILED_HEADER := ${PRECOMPILED_HEADER_SOURCE:%.hpp=%.hpp.gch}

OBJECTS := \
	${patsubst %.cpp,%.o,${filter %.cpp,$(SOURCES)}} \
	${patsubst %.c,%.o,${filter %.c,$(SOURCES)}} \
	${patsubst %.cu,%.o,${filter %.cu,$(SOURCES)}}


# Build rules
.SUFFIXES: .cu .o

all: $(TARGET)

$(TARGET) : $(OBJECTS)
	$(NVCC) -shared $(LDFLAGS) $(LIBS) $(OBJECTS) -o $@ 

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o : %.c $(PRECOMPILED_HEADER)
	$(CC) $(CFLAGS) -c $< -o $@

%.o : %.cpp $(PRECOMPILED_HEADER)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PRECOMPILED_HEADER) : $(PRECOMPILED_HEADER_SOURCE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $<

clean:
	rm -rf $(OBJECTS) $(TARGET)

