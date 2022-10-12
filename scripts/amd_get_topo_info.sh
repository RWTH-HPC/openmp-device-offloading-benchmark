#!/bin/bash

RES_DIR="$(pwd)/results"
mkdir -p ${RES_DIR}

sbatch --partition=mi210 --output=${RES_DIR}/topo_mi210.txt amd_get_topo_info.sbatch
sbatch --partition=mi250 --output=${RES_DIR}/topo_mi250.txt amd_get_topo_info.sbatch