#!/bin/bash -l

#$ -N AVDDM
#$ -l h_rt=2:00:00
#$ -l mem=4G
# no of cpu slots I am requesting 
#$ -pe smp 1

module purge
module load gcc-libs/4.9.2
module load python3/recommended
module load compilers/gnu/4.9.2
module load numactl/2.0.12
module load binutils/2.29.1/gnu-4.9.2
module load ucx/1.9.0/gnu-4.9.2
module load mpi/openmpi/4.1.1/gnu-4.9.2
module load mpi4py/3.1.4/gnu-4.9.2

module load numactl
module load binutils


pip install --upgrade pip
pip uninstall git+https://github.com/mwshinn/PyDDM.git@dev
pip install pathos
pip install psutil 

mpirun -np 1 python test_installs.py
