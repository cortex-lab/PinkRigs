#$ -N AVDDM
#$ -l h_rt=12:00:00
# no of cpu slots I am requesting 
#$ -pe mpi 2
#$ -l mem=4G

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
pip install git+https://github.com/mwshinn/PyDDM.git@dev
pip install pathos
pip install psutil 

mpirun -np 2 python DDMfit_mpi.py