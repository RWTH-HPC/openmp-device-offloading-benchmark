#!/bin/bash

sbatch --partition=mi210 --output=topo_mi210.txt amd_get_topo_info.sbatch
sbatch --partition=mi250 --output=topo_mi250.txt amd_get_topo_info.sbatch