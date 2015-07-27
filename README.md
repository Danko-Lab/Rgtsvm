# Rgtsvm
The SVM package for GPU architecture based on the GTSVM software(http://ttic.uchicago.edu/~cotter/projects/gtsvm/)

Now it works on stampede.tacc.xsede.org as the followings script:

\#loading boost, cuda and appropraite gcc
module load gcc/4.7.1
module load cuda
module load boost/1.55.0

$R CMD INSTALL --configure-args="--with-cuda-home=/opt/apps/cuda/5.5 --with-boost-home=/opt/apps/gcc4_7/boost/1.55.0" Rgtsvm
\#End
