#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --exclusive

echo "===== hostname"
hostname

echo "===== numactl -H"
numactl -H

echo "===== lscpu"
lscpu

echo "===== hwloc"
hwloc-ls

echo "===== GPU Info"
nvidia-smi
nvidia-smi topo -m
