#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}

# sbatch --partition=c16g --gres=gpu:pascal:2 --account=supp0001 --output=${RES_DIR}/topo_c16g.txt nvidia_get_topo_info.sbatch
# sbatch --partition=c18g --gres=gpu:2 --account=supp0001 --output=${RES_DIR}/topo_c18g.txt nvidia_get_topo_info.sbatch
sbatch --partition=c23g --gres=gpu:4 --exclusive --account=supp0001 --output=${RES_DIR}/topo_c23g.txt nvidia_get_topo_info.sbatch