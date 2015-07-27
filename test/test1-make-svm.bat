#!/bin/tcsh
#SBATCH -J svm.e1 # Job name
#SBATCH -o svm.e1.%j.out # stdout; %j expands to jobid
#SBATCH -e svm.e1.%j.err # stderr; skip to combine stdout and stderr
#SBATCH -p normal
#SBATCH -n 1 # one node and one task
#SBATCH -t 1:00:00 # max time
#SBATCH --mail-user=zw355@cornell.edu
#SBATCH --mail-type=ALL

$R --vanilla --no-save < test1-make-svm.r > test1-make-svm.out





