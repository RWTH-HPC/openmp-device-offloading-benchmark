#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}
COMMON_EXPORTS="GPU_ARCH,USE_CUDA,CPU_BIND"

####################################################
### OpenMP target based
####################################################
export USE_HIP=0

export GPU_ARCH=mi250
export CPU_BIND="--cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=mi250 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi250.txt amd_run_latency.sbatch

export GPU_ARCH=mi210
export CPU_BIND="--cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=mi210 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi210.txt amd_run_latency.sbatch

####################################################
### HIP based
####################################################
export USE_HIP=1

export GPU_ARCH=mi250
export CPU_BIND="--cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=mi250 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi250-hip.txt amd_run_latency.sbatch

export GPU_ARCH=mi210
export CPU_BIND="--cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=mi210 --ntasks-per-node=16 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_mi210-hip.txt amd_run_latency.sbatch