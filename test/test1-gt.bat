#!/bin/tcsh
#SBATCH -J svm.gt1 # Job name
#SBATCH -o svm.gt1.%j.out # stdout; %j expands to jobid
#SBATCH -e svm.gt1.%j.err # stderr; skip to combine stdout and stderr
#SBATCH -p gpu # queue or gpudev
#SBATCH -n 1 # one node and one task
#SBATCH -t 12:00:00 # max time
#SBATCH --mail-user=zw355@cornell.edu
#SBATCH --mail-type=ALL


module load cuda

$R --vanilla --no-save < rgtsvm-svc-tune.R > rgtsvm-svc-tune.out





