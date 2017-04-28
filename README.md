# Rgtsvm

The e1071 compatiblility SVM package for GPU architecture based on the GTSVM software (http://ttic.uchicago.edu/~cotter/projects/gtsvm/)

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


## Functions:

Firstly our implementation is encapsulated in one R package which is backwardscompatible with the e1071 implementation. 

The package has the following features:

> 1) Binary classification, multiclass classification and epsilon regression

> 2) 4 kernel functions (linear, polynomial, radial basis function and sigmoidal kernel)

> 3) K-fold cross validation 

> 4) Tuning parameters in kernel function or in SVM primary space or dual space (C, e).

> 5) Big matrix used in training or predicting data

> 6) Altering cost values for minor class or major class.

## Performance

![Image of comparison with e1071 and Rgtsvm ](https://github.com/Danko-Lab/Rgtsvm/blob/master/img/Rgtsvm_table.png)

![Image of comparison with e1071 and Rgtsvm ](https://github.com/Danko-Lab/Rgtsvm/blob/master/img/Rgtsvm_perf.png)

## Installation Instructions:

Rgtsvm is only available for the Linux and Mac OSX. The source code can be downloaded from this repository (https://github.com/Danko-Lab/Rgtsvm.git). 

### Required software and packages
    
1. R (http://www.r-project.org/)
    
2. CUDA library (https://github.com/Danko-Lab/dREG).
    
3. Boost library (https://github.com/arq5x/bedtools2/)
    
4. Extra R Package: bit64
    
### Install Rgtsvm

Please install the required R package before you install Rgtsvm package. After the  installation of `dREG`, `snowfall` and `data.table` package, please install the **Rgtsvm** as following steps.

```
git clone https://github.com/Danko-Lab/Rgtsvm.git

cd Rgtsvm

R CMD INSTALL --configure-args="--with-cuda-home=$CUDA_PATH --with-boost-home=$BOOST_PATH" Rgtsvm

```

##Usage instructions

Rgtsvm implement the following functions on GPU package(GTSVM)

> `svm`: a function to train a support vector machine by the C-classfication method and epsilon regression on GPU

> `predict`: a function to predict values based upon a model trained by `svm` in package Rgtsvm

> `tune`: a function to tune hyperparameters of statistical methods using a grid search over supplied parameter ranges

> `plot.tune`: visualizes the results of parameter tuning

> `load.svmlight`: a function to load SVMlight data file into a sparse matrix

Please check the details in the manual (https://github.com/Danko-Lab/Rgtsvm/blob/master/Rgtsvm-manual.pdf).

To use Rgtsvm, type: 

```
library(Rgtsvm);

?svm

model <- svm(Species ~ ., data = iris);
```

###Installation instructions on *stampede.tacc.xsede.org*
-----------

```
module load gcc/4.7.1
module load cuda
module load boost/1.55.0

R CMD INSTALL --configure-args="--with-cuda-home=/opt/apps/cuda/6.5 --with-boost-home=/opt/apps/gcc4_7/boost/1.55.0" Rgtsvm
```


###Installation instructions on *supermic.cct-lsu.xsede.org*
-----------

```
module load r
module load cuda/6.5
R CMD INSTALL --configure-args="--with-cuda-home=/usr/local/packages/cuda/6.5 --with-boost-home=/usr/local/packages/boost/1.55.0/INTEL-14.0.2-python-2.7.7-anaconda" Rgtsvm
```

