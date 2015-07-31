#!/bin/tcsh
#SBATCH -J svm.gt2 # Job name
#SBATCH -o svm.gt2.%j.out # stdout; %j expands to jobid
#SBATCH -e svm.gt2.%j.err # stderr; skip to combine stdout and stderr
#SBATCH -p gpudev # queue or gpudev
#SBATCH -n 1 # one node and one task
#SBATCH -t 4:00:00 # max time
#SBATCH --mail-user=zw355@cornell.edu
#SBATCH --mail-type=ALL


module load cuda

$R --vanilla --no-save < test2-gt.R > test2-gt.out





