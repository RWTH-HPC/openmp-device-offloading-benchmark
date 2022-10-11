#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}

export GPU_ARCH=sm_60
sbatch --partition=c16g --gres=gpu:pascal:2 --account=supp0001 --export=GPU_ARCH --output=${RES_DIR}/results_lat_c16g.txt nvidia_run_latency.sbatch
export GPU_ARCH=sm_70
sbatch --partition=c16g --gres=gpu:2 --account=supp0001 --export=GPU_ARCH --output=${RES_DIR}/results_lat_c18g.txt nvidia_run_latency.sbatch