#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}

# OpenMP target based
export USE_HIP=0
export GPU_ARCH=mi250
sbatch --partition=mi250 --export=GPU_ARCH,USE_HIP --output=${RES_DIR}/results_lat_mi250.txt amd_run_latency.sbatch
export GPU_ARCH=mi210
sbatch --partition=mi210 --export=GPU_ARCH,USE_HIP --output=${RES_DIR}/results_lat_mi210.txt amd_run_latency.sbatch

# HIP based
export USE_HIP=1
export GPU_ARCH=mi250
sbatch --partition=mi250 --export=GPU_ARCH,USE_HIP --output=${RES_DIR}/results_lat_mi250-hip.txt amd_run_latency.sbatch
export GPU_ARCH=mi210
sbatch --partition=mi210 --export=GPU_ARCH,USE_HIP --output=${RES_DIR}/results_lat_mi210-hip.txt amd_run_latency.sbatch