#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --exclusive

GPU_ARCH=${GPU_ARCH:-"sm_70"}

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== Experiments"
module purge
module load DEVELOP
module load clang/12
module load cuda/10.2

# switch to directory
cd ../benchmarks/latency
# clean first
TARGET_EXT=${GPU_ARCH} make distclean
# build app
TARGET_EXT=${GPU_ARCH} CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=${GPU_ARCH}" make
# run app
TARGET_EXT=${GPU_ARCH} make run