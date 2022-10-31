#include <cstdio>
#include <cstring>
#include <cfloat>

#include <cuda_runtime.h>
#include <omp.h>

#ifndef REPS
#define REPS 10
#endif

#ifndef INCLUDE_ALLOC
#define INCLUDE_ALLOC 1
#endif

__global__ void empty(size_t n, char * array) {
    // do nothing!
}

int main(int argc, char const * argv[]) {
    int ncores;
    int ndev;
    double *** bandwidth = NULL;
    double * min_bandwidth = NULL;

    const int nsizes = 3;
    size_t array_sizes_bytes[3] = {10000000, 100000000, 1000000000};
    const size_t MAX_BUF_SIZE = 1000000000;

    // Determine number of cores and devices.
    cudaGetDeviceCount(&ndev);
    ncores = omp_get_num_procs();

    // get representative data to fill device
    int max_threads_per_block = 0;
    int max_threads_per_mp = 0;
    int mp_count = 0;
    cudaDeviceGetAttribute (&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute (&max_threads_per_mp, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute (&mp_count, cudaDevAttrMultiProcessorCount, 0);
    int n_blocks_to_start = (max_threads_per_mp / max_threads_per_block) * mp_count;

    fprintf(stdout, "---------------------------------------------------------------\n");
    fprintf(stdout, "number of array sizes: %d\n", nsizes);
    fprintf(stdout, "number of cores:   %d\n", ncores);
    fprintf(stdout, "number of devices: %d\n", ndev);
    fprintf(stdout, "number of repetitions: %d\n", REPS);
    fprintf(stdout, "---------------------------------------------------------------\n");
    fprintf(stdout, "mp_count: %d\n", mp_count);
    fprintf(stdout, "max_threads_per_block: %d\n", max_threads_per_block);
    fprintf(stdout, "max_threads_per_mp: %d\n", max_threads_per_mp);
    fprintf(stdout, "n_blocks_to_start: %d\n", n_blocks_to_start);
    fprintf(stdout, "---------------------------------------------------------------\n");

    // Allocate the memory to store the result data.
    bandwidth = (double ***)malloc(nsizes * sizeof(double **));
    min_bandwidth = (double *)malloc(nsizes * sizeof(double));
    for (int s = 0; s < nsizes; s++) {
        bandwidth[s] = (double **)malloc(ncores * sizeof(double *));
        min_bandwidth[s] = DBL_MAX;
        for (int c = 0; c < ncores; c++) {
            bandwidth[s][c] = (double *)malloc(ndev * sizeof(double));
        }
    }

    /*// Print the OpenMP thread affinity info.
    #pragma omp parallel num_threads(ncores)
    {
        omp_display_affinity(NULL);
    }*/
    
    // Allocate per thread buffers
    char ** per_thread_buffs = (char **)malloc(ncores * sizeof(char *));
    #pragma omp parallel num_threads(ncores)
    {
        int cur_thread = omp_get_thread_num();
        per_thread_buffs[cur_thread] = (char *)malloc(MAX_BUF_SIZE);
        // init buffer using first-touch
        memset(per_thread_buffs[cur_thread], 0, MAX_BUF_SIZE);
    }

    fprintf(stdout, "---------------------------------------------------------------\n");

    // Perform some warm-up to make sure that all threads are up and running,
    // and the GPUs have been properly initialized.
    fprintf(stdout, "warm up...\n");
    #pragma omp parallel num_threads(ncores)
    {
        for (int c = 0; c < ncores; c++) {
            if (omp_get_thread_num() == c) {
                for (int d = 0; d < ndev; d++) {
                    cudaSetDevice(d);
                    empty<<<n_blocks_to_start, max_threads_per_block>>>(d, NULL);
                    cudaDeviceSynchronize();
                }
            }
            #pragma omp barrier
        }
    }
    fprintf(stdout, "---------------------------------------------------------------\n");

    // Perform the actual measurements.
    fprintf(stdout, "measurements...\n");
    #pragma omp parallel num_threads(ncores)
    {
        int cur_thread = omp_get_thread_num();
        for (int c = 0; c < ncores; c++) {
            if (cur_thread == c) {
                for (int s = 0; s < nsizes; s++) {
                    size_t cur_size     = array_sizes_bytes[s];
                    double tmp_size_mb  = ((double)cur_size / 1e6);

                    for (int d = 0; d < ndev; d++) {
                        fprintf(stdout, "running for thread=%3d, size=%7.2fMB and device=%2d\n", c, tmp_size_mb, d);
                        fflush(stdout);
                        cudaSetDevice(d);
                        
                        // allocate and initialize data once (first-touch)
                        // char * buffer = (char *)malloc(cur_size);
                        // memset(buffer, 0, cur_size);
                        char * buffer_dev = NULL;
                        char * buffer = per_thread_buffs[cur_thread];
#if !INCLUDE_ALLOC
                        cudaMalloc(&buffer_dev, sizeof(*buffer_dev) * cur_size);
#endif
                        double ts = omp_get_wtime();
                        for (int r = 0; r < REPS; r++) {
#if INCLUDE_ALLOC
                            cudaMalloc(&buffer_dev, sizeof(*buffer_dev) * cur_size);
#endif
                            cudaMemcpy(buffer_dev, buffer, sizeof(*buffer) * cur_size, cudaMemcpyHostToDevice);
                            empty<<<n_blocks_to_start, max_threads_per_block>>>(cur_size, buffer_dev);
                            cudaMemcpy(buffer, buffer_dev, sizeof(*buffer) * cur_size, cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
#if INCLUDE_ALLOC
                            cudaFree(buffer_dev);
#endif
                        }
                        double te = omp_get_wtime();
                        double avg_time_sec = (te - ts) / ((double) REPS);
                        bandwidth[s][c][d] = tmp_size_mb * 2 / avg_time_sec;
                        if(bandwidth[s][c][d] < min_bandwidth[s]) {
                            min_bandwidth[s] = bandwidth[s][c][d];
                        }

#if !INCLUDE_ALLOC
                        cudaFree(buffer_dev);
#endif
                    }
                }
            }
            #pragma omp barrier
        }
    }
    fprintf(stdout, "---------------------------------------------------------------\n");

    // free memory and cleanup
    for(int i = 0; i < ncores; i++) {
        free(per_thread_buffs[i]);
    }
    free(per_thread_buffs);

    fprintf(stdout, "---------------------------------------------------------------\n");
    fprintf(stdout, "Absolute measurements (MB/s)\n");
    fprintf(stdout, "---------------------------------------------------------------\n");
    for (int s = 0; s < nsizes; s++) {
        size_t cur_size = array_sizes_bytes[s];
        fprintf(stdout, "##### Problem Size: %.2f KB\n", cur_size / 1000.0);
        fprintf(stdout, ";");
        for (int c = 0; c < ncores; c++) {
            fprintf(stdout, "Core %d%c", c, c<ncores-1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++) {
            fprintf(stdout, "GPU %d;", d);
            for (int c = 0; c < ncores; c++) {
                fprintf(stdout, "%lf%c", bandwidth[s][c][d], c<ncores-1 ? ';' : '\n');
            }
        }
    }
    fprintf(stdout, "\n\n");
    for (int c = 0; c < ncores; c++) {
        fprintf(stdout, "##### Core: %d\n", c);
        fprintf(stdout, ";");
        for (int s = 0; s < nsizes; s++) {
            size_t cur_size = array_sizes_bytes[s];
            fprintf(stdout, "%.2f KB%c", cur_size / 1000.0, s<nsizes-1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++) {
            fprintf(stdout, "GPU %d;", d);
            for (int s = 0; s < nsizes; s++) {
                fprintf(stdout, "%lf%c", bandwidth[s][c][d], s<nsizes-1 ? ';' : '\n');
            }
        }
    }

    fprintf(stdout, "---------------------------------------------------------------\n");
    fprintf(stdout, "Relative measurements to minimum bandwidth for size\n");
    fprintf(stdout, "---------------------------------------------------------------\n");
    for (int s = 0; s < nsizes; s++) {
        size_t cur_size = array_sizes_bytes[s];
        fprintf(stdout, "##### Problem Size: %.2f KB\n", cur_size / 1000.0);
        fprintf(stdout, ";");
        for (int c = 0; c < ncores; c++) {
            fprintf(stdout, "Core %d%c", c, c<ncores-1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++) {
            fprintf(stdout, "GPU %d;", d);
            for (int c = 0; c < ncores; c++) {
                fprintf(stdout, "%lf%c", bandwidth[s][c][d] / min_bandwidth[s], c<ncores-1 ? ';' : '\n');
            }
        }
    }
    fprintf(stdout, "\n\n");
    for (int c = 0; c < ncores; c++) {
        fprintf(stdout, "##### Core: %d\n", c);
        fprintf(stdout, ";");
        for (int s = 0; s < nsizes; s++) {
            size_t cur_size = array_sizes_bytes[s];
            fprintf(stdout, "%.2f KB%c", cur_size / 1000.0, s<nsizes-1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++) {
            fprintf(stdout, "GPU %d;", d);
            for (int s = 0; s < nsizes; s++) {
                fprintf(stdout, "%lf%c", bandwidth[s][c][d] / min_bandwidth[s], s<nsizes-1 ? ';' : '\n');
            }
        }
    }

    return 0;
}