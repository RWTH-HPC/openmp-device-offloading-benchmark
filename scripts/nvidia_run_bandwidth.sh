#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}
COMMON_EXPORTS="GPU_ARCH,USE_CUDA,INCLUDE_ALLOC,MPICMD"

####################################################
### OpenMP target based
####################################################
export USE_CUDA=0

# CLAIX 2016
# export GPU_ARCH=sm_60
# export MPICMD="srun --cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c16g --ntasks-per-node=12 --gres=gpu:pascal:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c16g.txt nvidia_run_bandwidth.sbatch

# CLAIX 2018
# export GPU_ARCH=sm_70
# export MPICMD="srun --cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c18g --ntasks-per-node=12 --gres=gpu:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c18g.txt nvidia_run_bandwidth.sbatch

# CLAIX 2023
export GPU_ARCH=sm_90
export MPICMD="srun --cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=c23g --ntasks-per-node=16 --gres=gpu:4 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c23g.txt nvidia_run_bandwidth.sbatch

####################################################
### CUDA based (w/ alloc included)
####################################################
export USE_CUDA=1
export INCLUDE_ALLOC=1

# CLAIX 2016
# export GPU_ARCH=sm_60
# export MPICMD="srun --cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c16g --ntasks-per-node=12 --gres=gpu:pascal:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c16g-cuda.txt nvidia_run_bandwidth.sbatch

# CLAIX 2018
# export GPU_ARCH=sm_70
# export MPICMD="srun --cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c18g --ntasks-per-node=12 --gres=gpu:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c18g-cuda.txt nvidia_run_bandwidth.sbatch

# CLAIX 2023
export GPU_ARCH=sm_90
export MPICMD="srun --cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=c23g --ntasks-per-node=16 --gres=gpu:4 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c23g-cuda.txt nvidia_run_bandwidth.sbatch

####################################################
### CUDA based (w/o alloc included)
####################################################
export USE_CUDA=1
export INCLUDE_ALLOC=0

# CLAIX 2016
# export GPU_ARCH=sm_60
# export MPICMD="srun --cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c16g --ntasks-per-node=12 --gres=gpu:pascal:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c16g-cuda-noalloc.txt nvidia_run_bandwidth.sbatch

# CLAIX 2018
# export GPU_ARCH=sm_70
# export MPICMD="srun --cpu-bind=map_ldom:0,0,0,1,1,1,2,2,2,3,3,3"
# sbatch --partition=c18g --ntasks-per-node=12 --gres=gpu:2 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c18g-cuda-noalloc.txt nvidia_run_bandwidth.sbatch

# CLAIX 2023
export GPU_ARCH=sm_90
export MPICMD="srun --cpu-bind=map_ldom:0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
sbatch --partition=c23g --ntasks-per-node=16 --gres=gpu:4 --account=supp0001 --export=${COMMON_EXPORTS} --output=${RES_DIR}/results_bw_c23g-cuda-noalloc.txt nvidia_run_bandwidth.sbatch
