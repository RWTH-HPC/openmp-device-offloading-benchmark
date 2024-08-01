#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}
COMMON_EXPORTS="GPU_ARCH,USE_HIP,INCLUDE_ALLOC,MPICMD"
export MPICMD="mpirun --map-by ppr:2:numa --oversubscribe"

####################################################
### OpenMP target based
####################################################
export USE_HIP=0

export GPU_ARCH=mi250
sbatch --partition=mi250 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_mi250.txt amd_run_bandwidth.sbatch

export GPU_ARCH=mi210
sbatch --partition=mi210 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_mi210.txt amd_run_bandwidth.sbatch

####################################################
### HIP based (w/ alloc included)
####################################################
# export USE_HIP=1
# export INCLUDE_ALLOC=1

# export GPU_ARCH=mi250
# sbatch --partition=mi250 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_mi250-hip_incl-alloc.txt amd_run_bandwidth.sbatch

# export GPU_ARCH=mi210
# sbatch --partition=mi210 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_mi210-hip_incl-alloc.txt amd_run_bandwidth.sbatch

####################################################
### HIP based (w/o alloc included)
####################################################
export USE_HIP=1
export INCLUDE_ALLOC=0

export GPU_ARCH=mi250
sbatch --partition=mi250 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_mi250-hip.txt amd_run_bandwidth.sbatch

export GPU_ARCH=mi210
sbatch --partition=mi210 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_mi210-hip.txt amd_run_bandwidth.sbatch
