# openmp-device-affinity
Experiments and benchmarks for evaluating device affinity in OpenMP

## 1. Latency & bandwidth experiments

### 1.1 NVIDIA GPUs
To execute with Clang and NVIDIA GPUs execute the following:
```bash
# switch to target directory
cd benchmarks/latency
# -or-
cd benchmarks/bandwidth

# make sure Clang and CUDA are loaded (depends on cluster/site environment)
# Here: RWTH Aachen University
module purge
module load DEVELOP
module load clang/12
module load cuda/10.2

# build program
# Note: Make sure to choose the correct architecture version for your GPU (e.g. sm_70 for Volta, sm_60 for Pascal)
CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70" make

# run experiments with
make run &> results.txt
# -or-
make run 2>&1 | tee results.txt
```

### 1.2 AMD GPUs
To execute with Clang and AMD GPUs execute the following:
```bash
# switch to target directory
cd benchmarks/latency
# -or-
cd benchmarks/bandwidth

# build program
CC=amdclang CCFLAGS="-O3 -std=gnu99 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a" make

# run experiments with
make run &> results.txt
# -or-
make run 2>&1 | tee results.txt
```
