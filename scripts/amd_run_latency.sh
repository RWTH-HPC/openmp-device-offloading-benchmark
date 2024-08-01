#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}
COMMON_EXPORTS="GPU_ARCH,USE_CUDA,MPICMD"

####################################################
### OpenMP target based
####################################################
export USE_HIP=0

export GPU_ARCH=mi250
export MPICMD="mpirun -np 16 --map-by ppr:2:numa --oversubscribe"
sbatch --partition=mi250 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi250.txt amd_run_latency.sbatch

export GPU_ARCH=mi210
export MPICMD="mpirun -np 16 --map-by ppr:2:numa --oversubscribe"
sbatch --partition=mi210 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi210.txt amd_run_latency.sbatch

####################################################
### HIP based
####################################################
export USE_HIP=1

export GPU_ARCH=mi250
export MPICMD="mpirun -np 16 --map-by ppr:2:numa --oversubscribe"
sbatch --partition=mi250 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi250-hip.txt amd_run_latency.sbatch

export GPU_ARCH=mi210
export MPICMD="mpirun -np 16 --map-by ppr:2:numa --oversubscribe"
sbatch --partition=mi210 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi210-hip.txt amd_run_latency.sbatch