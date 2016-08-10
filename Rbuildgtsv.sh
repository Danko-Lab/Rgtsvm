module load gcc/4.7.1
module load cuda/6.5
module load boost/1.55.0

$R CMD INSTALL --configure-args="--with-cuda-home=/opt/apps/cuda/6.5 --with-boost-home=/opt/apps/gcc4_7/boost/1.55.0" Rgtsvm
