#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --exclusive

GPU_ARCH=${GPU_ARCH:-"mi250"}
USE_HIP=${USE_HIP:-0}
CPU_BIND=${CPU_BIND:-"--distribution=block:cyclic"}

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== Experiments"
# switch to directory
cd ../benchmarks/latency

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false

if [[ "${USE_HIP}" = "1" ]]
then
    # clean first
    TARGET_EXT=${GPU_ARCH} make -f Makefile.hip clean
    # build app
    TARGET_EXT=${GPU_ARCH} make -f Makefile.hip REPS=1000
    # run app
    TARGET_EXT=${GPU_ARCH} srun ${CPU_BIND} make -f Makefile.hip run
else
    # clean first
    TARGET_EXT=${GPU_ARCH} make clean
    # build app
    TARGET_EXT=${GPU_ARCH} CC=amdclang CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a" make
    # run app
    TARGET_EXT=${GPU_ARCH} srun ${CPU_BIND} make run
fi