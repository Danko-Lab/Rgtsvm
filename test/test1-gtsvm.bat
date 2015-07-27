#!/bin/tcsh
#SBATCH -J svm.adult.teste # Job name
#SBATCH -o svm.adult.test.%j.out # stdout; %j expands to jobid
#SBATCH -e svm.adult.test.%j.err # stderr; skip to combine stdout and stderr
#SBATCH -p gpudev # queue
#SBATCH -n 1 # one node and one task
#SBATCH -t 1:00:00 # max time
#SBATCH --mail-user=zw355@cornell.edu
#SBATCH --mail-type=ALL


module load cuda

/home1/03350/tg826494/Rsrc/GTSVM/gtsvm/bin/gtsvm_initialize -f test1-gt.dat -o test1-gt.mdl -C 1 -k gaussian -1 0.05

/home1/03350/tg826494/Rsrc/GTSVM/gtsvm/bin/gtsvm_optimize -i test1-gt.mdl -o test1-gt.mdl -e 0.001 -n 100000000

/home1/03350/tg826494/Rsrc/GTSVM/gtsvm/bin/gtsvm_shrink -i test1-gt.mdl -o test1-gt-shrunk.mdl

/home1/03350/tg826494/Rsrc/GTSVM/gtsvm/bin/gtsvm_classify -f test1-gt.dat -i test1-gt-shrunk.mdl -o test1-gt-testing.txt

