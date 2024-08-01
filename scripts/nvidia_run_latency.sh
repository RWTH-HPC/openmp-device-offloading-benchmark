#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}
COMMON_EXPORTS="GPU_ARCH,USE_CUDA,CPU_BIND"

####################################################
### OpenMP target based
####################################################
export USE_CUDA=0

# CLAIX 2016
# export GPU_ARCH=sm_60
# export CPU_BIND="--cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c16g --ntasks-per-node=12 --gres=gpu:pascal:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_c16g.txt nvidia_run_latency.sbatch

# CLAIX 2018
# export GPU_ARCH=sm_70
# export CPU_BIND="--cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c18g --ntasks-per-node=12 --gres=gpu:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_c18g.txt nvidia_run_latency.sbatch

# CLAIX 2023
export GPU_ARCH=sm_90
export CPU_BIND="--cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=c23g --ntasks-per-node=16 --gres=gpu:4 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_c23g.txt nvidia_run_latency.sbatch

####################################################
### CUDA based
####################################################
export USE_CUDA=1

# CLAIX 2016
# export GPU_ARCH=sm_60
# export CPU_BIND="--cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c16g --ntasks-per-node=12 --gres=gpu:pascal:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_c16g-cuda.txt nvidia_run_latency.sbatch

# CLAIX 2018
# export GPU_ARCH=sm_70
# export CPU_BIND="--cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c18g --ntasks-per-node=12 --gres=gpu:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_c18g-cuda.txt nvidia_run_latency.sbatch

# CLAIX 2023
export GPU_ARCH=sm_90
export CPU_BIND="--cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=c23g --ntasks-per-node=16 --gres=gpu:4 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_lat_c23g-cuda.txt nvidia_run_latency.sbatch