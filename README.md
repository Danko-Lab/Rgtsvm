# Rgtsvm

The e1071 compatibility SVM package for GPU architecture based on the [GTSVM](http://ttic.uchicago.edu/~cotter/projects/gtsvm/) software.

[Vignette (https://github.com/Danko-Lab/Rgtsvm/blob/master/Rgtsvm-vignette.pdf)](https://github.com/Danko-Lab/Rgtsvm/blob/master/Rgtsvm-vignette.pdf).

## Intoduction:

SVM is a popular and powerful machine learning method for classification, regression, and other learning tasks. In the R community, many users use the e1071 package, which offers an interface to the
C++ implementation of libsvm, featuring with C classification, epsilon regression, one class classification, eregression, v regression, cross validation, parameter tuning and four kernels (linear, polynomial, radial
basis function, and sigmoidal kernels formula). Although this implementation is widely used, it is not sufficiently fast to handle largescale classification or regression tasks.

To improve the performance, we have recently implemented SVMs on a graphical processing unit (GPU). GPUs are a massively parallel execution environment that provide many advantages when computing SVMs: 

> 1st: a large number of independent threads build a highly parallel and fast computational engine; 

> 2nd: using GPU Basic Linear Algebra Subroutines (CUBLAS) instead of conventional Intel Math Kernel Library (MKL) can speed up the application 3 to 5 times; 

> 3rd: kernel functions called for the huge samples will be more efficient on SIMD (Single Instruction Multiple Data) computer. 

GPU tools dedicated to SVMs have recently been developed and provide command line interface and binary classification, which functions are comparable to the e1071 package. Among these SVM programs, GT SVM ( Cotter, Srebro, and Keshet 2011) takes full advantage of GPU architecture and efficiently handles
sparse datasets through the use of a clustering technique. GT SVM is also implemented in C/C++ and provides simple functions that can make use of the package as a library. To enable the use of GT SVM without expertise in C/ C++, we implemented an R interface to GT SVM that combines the easeofuse of e1071 and the speed of the GT SVM GPU implementation. Our implementation consists of the
following: 

> 1) an R wrapper for GT SVM, 

> 2) matching the SVM interface in the e1071 package so that R written around each implementation is exchangeable, 

> 3) adding or altering some features in R code, such as cross-validation which is implemented in C/C++ in e1071 and has not been implemented in GT SVM.

> 4) Supporting prediction using multiple GPU cards on single host for further speedup computation.  


## Functions:

Firstly our implementation is encapsulated in one R package which is backwardscompatible with the e1071 implementation. 

The package has the following features:

> 1) Binary classification, multiclass classification and epsilon regression

> 2) 4 kernel functions (linear, polynomial, radial basis function and sigmoidal kernel)

> 3) K-fold cross validation 

> 4) Tuning parameters in kernel function or in SVM primary space or dual space (C, e).

> 5) Big matrix used in training or predicting data

> 6) Altering cost values for minor class or major class.

## Usage instructions

Rgtsvm implement the following functions on GPU package(GTSVM)

> `svm`: a function to train a support vector machine by the C-classfication method and epsilon regression on GPU

> `predict`: a function to predict values based upon a model trained by `svm` in package Rgtsvm

> `tune`: a function to tune hyperparameters of statistical methods using a grid search over supplied parameter ranges

> `plot.tune`: visualizes the results of parameter tuning

> `load.svmlight`: a function to load SVMlight data file into a sparse matrix

Please check the details in the ***manual*** (https://github.com/Danko-Lab/Rgtsvm/blob/master/Rgtsvm-manual.pdf) or the ***vignette***  (https://github.com/Danko-Lab/Rgtsvm/blob/master/Rgtsvm-vignette.pdf).


To use Rgtsvm, type: 

```
> library(Rgtsvm);

> ?svm

> model <- svm(Species ~ ., data = iris);
```

## Performance

![Image of comparison with e1071 and Rgtsvm ](https://github.com/Danko-Lab/Rgtsvm/blob/master/img/Rgtsvm_table.png)

![Image of comparison with e1071 and Rgtsvm ](https://github.com/Danko-Lab/Rgtsvm/blob/master/img/Rgtsvm_perf.png)

## Installation Instructions:

Rgtsvm is only available for the Linux and Mac OSX. The source code can be downloaded from this repository (https://github.com/Danko-Lab/Rgtsvm.git). 

### Required software and packages
    
1. R (http://www.r-project.org/)
    
2. CUDA library (https://developer.nvidia.com/cuda-toolkit-archive).
    
3. Boost library (http://www.boost.org/users/download/)
    
4. Extra R Package: bit64, snow, SparseM, Matrix
    
### Install Rgtsvm

Please install the required R package before you install Rgtsvm package. After the  installation of `bit64`, `snow`, `SparseM` and `Matrix` package, please install the **Rgtsvm** as following steps.

```

# Set $YOUR_CUDA_HOME and $YOUR_BOOST_HOME before installation

$ git clone https://github.com/Danko-Lab/Rgtsvm.git

$ cd Rgtsvm

$ make R_dependencies

$ R CMD INSTALL --configure-args="--with-cuda-home=$YOUR_CUDA_HOME --with-boost-home=$YOUR_BOOST_HOME" Rgtsvm

```

If you have installed the pakacge devtools, you can try these commands in R console:

```
> library(devtools)
> install_github("Danko-lab/Rgtsvm/Rgtsvm", args="--configure-args='--with-cuda-home=YOUR_CUDA_PATH --with-boost-home=YOU_BOOST_PATH'" )
```

Please check the ***vignette*** (https://github.com/Danko-Lab/Rgtsvm/blob/master/Rgtsvm-vignette.pdf) to see more details.

### Compile Rgtsvm using *CUDA 9.0*

CUDA 9.0 prohibits the architecture sm_20, which is the most early type for GeForce series. Please check this link.
http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

As described in above link, you should select one architecture type, such as sm_50 for TitanX or sm_60 for P100. And then  change the architecture type manually in the configure files, e.g.

$Rgtsvm\configure line 2381: NCFLAGS="-arch=sm_20 -O2"  --> NCFLAGS="-arch=sm_60 -O2"

$Rgtsvm\configure.ac line 30: NCFLAGS="-arch=sm_20 -O2" --> NCFLAGS="-arch=sm_60 -O2"

### Specify CUDA arch on CUDA 7 and later

CUDA 7.0 and later supports multiple NVIDIA GPU architectures that the CUDA files will be compiled for.
http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

As described in above link, you should select one architecture type, such as sm_50 for TitanX or sm_60 for P100, sm_35 for Tesla K40. And then set the architecture in your install command

```
> library(devtools)
> install_github("Danko-lab/Rgtsvm/Rgtsvm", args="--configure-args='--with-cuda-arch=sm_35 --with-cuda-home=YOUR_CUDA_PATH --with-boost-home=YOU_BOOST_PATH'" )
```
Or

```
 R CMD INSTALL --configure-args="-with-cuda-arch=sm_35 --with-cuda-home=YOUR_CUDA_PATH --with-boost-home=YOU_BOOST_PATH" Rgtsvm
```

### Installation Example

#### Installation instructions on *stampede.tacc.xsede.org*

```
module load gcc/4.7.1
module load cuda
module load boost/1.55.0

R CMD INSTALL --configure-args="--with-cuda-home=/opt/apps/cuda/6.5 --with-boost-home=/opt/apps/gcc4_7/boost/1.55.0" Rgtsvm
```

#### Installation instructions on *supermic.cct-lsu.xsede.org*

```
module load r
module load cuda/6.5
R CMD INSTALL --configure-args="--with-cuda-home=/usr/local/packages/cuda/6.5 --with-boost-home=/usr/local/packages/boost/1.55.0/INTEL-14.0.2-python-2.7.7-anaconda" Rgtsvm
```

## License (GPLv3)

Copyright(c) 2017 Zhong Wang

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

## How to cite

Wang, Z., Chu, T., Choate, L. A., & Danko, C. G. (2017). [Rgtsvm: Support Vector Machines on a GPU in R.](https://arxiv.org/abs/1706.05544) arXiv preprint arXiv:1706.05544.
