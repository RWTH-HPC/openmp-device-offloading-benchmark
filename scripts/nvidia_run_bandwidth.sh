#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}

# OpenMP target based
export USE_CUDA=0
export GPU_ARCH=sm_60
sbatch --partition=c16g --gres=gpu:pascal:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_bw_c16g.txt nvidia_run_bandwidth.sbatch
export GPU_ARCH=sm_70
sbatch --partition=c16g --gres=gpu:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA --output=${RES_DIR}/results_bw_c18g.txt nvidia_run_bandwidth.sbatch

# CUDA based (w/ alloc included)
export USE_CUDA=0
export INCLUDE_ALLOC=1
export GPU_ARCH=sm_60
sbatch --partition=c16g --gres=gpu:pascal:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA,INCLUDE_ALLOC --output=${RES_DIR}/results_bw_c16g-cuda.txt nvidia_run_bandwidth.sbatch
export GPU_ARCH=sm_70
sbatch --partition=c16g --gres=gpu:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA,INCLUDE_ALLOC --output=${RES_DIR}/results_bw_c18g-cuda.txt nvidia_run_bandwidth.sbatch

# CUDA based (w/o alloc included)
export USE_CUDA=0
export INCLUDE_ALLOC=0
export GPU_ARCH=sm_60
sbatch --partition=c16g --gres=gpu:pascal:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA,INCLUDE_ALLOC --output=${RES_DIR}/results_bw_c16g-cuda-noalloc.txt nvidia_run_bandwidth.sbatch
export GPU_ARCH=sm_70
sbatch --partition=c16g --gres=gpu:2 --account=supp0001 --export=GPU_ARCH,USE_CUDA,INCLUDE_ALLOC --output=${RES_DIR}/results_bw_c18g-cuda-noalloc.txt nvidia_run_bandwidth.sbatch


