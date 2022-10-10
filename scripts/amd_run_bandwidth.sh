#!/bin/bash

sbatch --partition=mi250 --output=results_bw_mi250.txt amd_run_bandwidth.sbatch
#sbatch --partition=mi210 --output=results_bw_mi210.txt amd_run_bandwidth.sbatch
