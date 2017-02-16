## This scipt is a modification of the configure.in file
## from the magma package (version 0.2.1)
## http://cran.r-project.org/web/packages/magma/
## The copyright information for the magma package is as follows:
## Copyright (C) 2010 Brian J Smith
## Licensed under GNU GPL 3 or later

# Compilers
EXT := cu

CXX := g++
NVCC := /usr/local/cuda-8.0/bin/nvcc
CXXFLAGS := -O2 -funroll-loops -fomit-frame-pointer -ffast-math -I/usr/local/cuda-8.0/include -I/usr/include -I/programs/R-3.2.5/include -Wall -fPIC -D_GNU_SOURCE -m64
NVCCFLAGS := -arch=sm_20 -O2 -I/usr/local/cuda-8.0/include -I/usr/include -DBOOST_NOINLINE='__attribute__((noinline))' -Xcompiler "-O2 -fpic -I/programs/R-3.2.5/include"
CFLAGS := -std=gnu99 -O2 -funroll-loops -fomit-frame-pointer -ffast-math -I/usr/local/cuda-8.0/include -I/usr/include -I/programs/R-3.2.5/include -Wall -fPIC -D_GNU_SOURCE -m64
NCFLAGS := -arch=sm_20 -O2 -I/usr/local/cuda-8.0/include -I/usr/include -DBOOST_NOINLINE='__attribute__((noinline))' -Xcompiler "-O2 -fpic -I/programs/R-3.2.5/include"
LDFLAGS := -L/usr/local/cuda-8.0/lib64
LIBS := -lcuda -lcublas

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
	$(NVCC) -shared $(LDFLAGS) $(LIB) $(OBJECTS) -o $@ 

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

