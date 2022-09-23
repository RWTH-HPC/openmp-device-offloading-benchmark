#!/bin/bash

sbatch --partition=c16g --gres=gpu:pascal:2 --account=supp0001 --output=topo_c16g.txt nvidia_get_topo_info.sbatch
sbatch --partition=c16g --gres=gpu:2 --account=supp0001 --output=topo_c18g.txt nvidia_get_topo_info.sbatch