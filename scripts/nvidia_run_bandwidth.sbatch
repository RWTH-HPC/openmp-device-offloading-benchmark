#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive

GPU_ARCH=${GPU_ARCH:-"sm_70"}
USE_CUDA=${USE_CUDA:-0}
INCLUDE_ALLOC=${INCLUDE_ALLOC:-1}

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== Experiments"
module purge
module load GCCcore/.12.3.0
module load Clang/16.0.6-CUDA-12.1.1

# switch to directory
cd ../benchmarks/bandwidth

export OMP_PLACES=`numactl -H | grep cpus | awk '(NF>3) {for (i = 4; i <= NF; i++) printf "%d,", $i}' | sed 's/.$//'`
export OMP_PROC_BIND=close

if [[ "${USE_CUDA}" = "1" ]]
then
    # clean first
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda clean
    # build app
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda INCLUDE_ALLOC=${INCLUDE_ALLOC}
    # run app
    TARGET_EXT=${GPU_ARCH} make -f Makefile.cuda run_no_numa
else
    # clean first
    TARGET_EXT=${GPU_ARCH} make clean
    # build app
    TARGET_EXT=${GPU_ARCH} CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=${GPU_ARCH}" make
    # run app
    TARGET_EXT=${GPU_ARCH} make run_no_numa
fi