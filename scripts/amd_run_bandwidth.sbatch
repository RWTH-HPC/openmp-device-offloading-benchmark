#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --exclusive

GPU_ARCH=${GPU_ARCH:-"mi250"}

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== Experiments"
# switch to directory
cd ../benchmarks/bandwidth
# clean first
TARGET_EXT=${GPU_ARCH} make clean
# build app
TARGET_EXT=${GPU_ARCH} CC=amdclang CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a" make
# run app
TARGET_EXT=${GPU_ARCH} make run