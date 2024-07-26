#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive

GPU_ARCH=${GPU_ARCH:-"sm_70"}
USE_CUDA=${USE_CUDA:-0}

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== Experiments"
module purge
module load iimpi/2023a
module load Clang/18.1.2-CUDA-12.3.0

# switch to directory
cd ../benchmarks/latency

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_CC=${CC}

if [[ "${USE_CUDA}" = "1" ]]
then
    # clean first
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda clean
    # build app
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda
    # run app
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda run_no_numa
else
    # clean first
    TARGET_EXT=${GPU_ARCH} make clean
    # build app
    TARGET_EXT=${GPU_ARCH} CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=${GPU_ARCH}" make
    # run app
    TARGET_EXT=${GPU_ARCH} srun --distribution=block:block ${FLAGS_MPI_BATCH} make run_no_numa
fi
