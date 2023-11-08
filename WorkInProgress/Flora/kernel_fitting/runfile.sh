#!/bin/bash -l


# array job style script

#$ -N kernelFit
#$ -l h_rt=12:00:00
#$ -l mem=8G

# 11 dataset,15 models/set
#$ -t 1-1

module purge
module load gcc-libs/4.9.2
module load python3/3.7 #appears crucial for xarray
module load compilers/gnu/4.9.2
module load numactl/2.0.12
module load binutils/2.29.1/gnu-4.9.2
module load ucx/1.9.0/gnu-4.9.2
module load mpi/openmpi/4.1.1/gnu-4.9.2
module load mpi4py/3.1.4/gnu-4.9.2

module load numactl
module load binutils

pip install  xarray



python kernel_fit_array.py $SGE_TASK_ID