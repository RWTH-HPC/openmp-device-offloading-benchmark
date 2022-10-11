#!/bin/bash

sbatch --partition=mi250 --output=results_lat_mi250.txt amd_run_latency.sbatch
sbatch --partition=mi210 --output=results_lat_mi210.txt amd_run_latency.sbatch