#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive

GPU_ARCH=${GPU_ARCH:-"sm_70"}
USE_CUDA=${USE_CUDA:-0}
INCLUDE_ALLOC=${INCLUDE_ALLOC:-1}
MPICMD=${MPICMD:-"srun --distribution=block:cyclic"}

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== Experiments"
module purge
module load iimpi/2023a
module load Clang/18.1.2-CUDA-12.3.0

# switch to directory
cd ../benchmarks/bandwidth

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false

if [[ "${USE_CUDA}" = "1" ]]
then
    # clean first
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda clean
    # build app
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda INCLUDE_ALLOC=${INCLUDE_ALLOC}
    # run app
    TARGET_EXT=${GPU_ARCH} ${MPICMD} make -f Makefile.cuda run
else
    # clean first
    TARGET_EXT=${GPU_ARCH} make clean
    # build app
    TARGET_EXT=${GPU_ARCH} CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=${GPU_ARCH}" make
    # run app
    TARGET_EXT=${GPU_ARCH} ${MPICMD} make run
fi