#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}

####################################################
### OpenMP target based
####################################################
export USE_CUDA=0

# CLAIX 2016
export GPU_ARCH=sm_60
sbatch --partition=c16g --ntasks-per-node=24 --gres=gpu:pascal:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_lat_c16g.txt nvidia_run_latency.sbatch

# CLAIX 2018
export GPU_ARCH=sm_70
sbatch --partition=c18g --ntasks-per-node=48 --gres=gpu:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_lat_c18g.txt nvidia_run_latency.sbatch

# CLAIX 2023
export GPU_ARCH=sm_90
sbatch --partition=c23g --ntasks-per-node=96 --gres=gpu:4 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_lat_c23g.txt nvidia_run_latency.sbatch

####################################################
### CUDA based
####################################################
export USE_CUDA=1

# CLAIX 2016
export GPU_ARCH=sm_60
sbatch --partition=c16g --gres=gpu:pascal:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_lat_c16g-cuda.txt nvidia_run_latency.sbatch

# CLAIX 2018
export GPU_ARCH=sm_70
sbatch --partition=c18g --gres=gpu:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_lat_c18g-cuda.txt nvidia_run_latency.sbatch

# CLAIX 2023
export GPU_ARCH=sm_90
sbatch --partition=c23g --gres=gpu:4 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_lat_c23g-cuda.txt nvidia_run_latency.sbatch