#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --exclusive

GPU_ARCH=${GPU_ARCH:-"mi250"}
USE_HIP=${USE_HIP:-0}
INCLUDE_ALLOC=${INCLUDE_ALLOC:-1}
MPICMD=${MPICMD:-"srun --distribution=block:cyclic"}

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== Experiments"
module load aomp/17.0-0
module load openmpi/aomp/4.1.1
export MPICC=mpicc

# switch to directory
cd ../benchmarks/bandwidth

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false

if [[ "${USE_HIP}" = "1" ]]
then
    # clean first
    TARGET_EXT=${GPU_ARCH} make -f Makefile.hip clean
    # build app
    TARGET_EXT=${GPU_ARCH} make -f Makefile.hip INCLUDE_ALLOC=${INCLUDE_ALLOC}
    # run app
    TARGET_EXT=${GPU_ARCH} ${MPICMD} make -f Makefile.hip run
else
    # clean first
    TARGET_EXT=${GPU_ARCH} make clean
    # build app
    TARGET_EXT=${GPU_ARCH} CC=amdclang CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -Wno-error=implicit-function-declaration" make
    # run app
    TARGET_EXT=${GPU_ARCH} ${MPICMD} make run
fi