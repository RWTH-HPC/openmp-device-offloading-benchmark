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

echo "===== GPU Info"
rocm-smi --showhw
rocm-smi --showproductname
rocm-smi --showtopo