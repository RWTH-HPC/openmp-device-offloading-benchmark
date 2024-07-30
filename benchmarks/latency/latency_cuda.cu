#include "../common/util.h"
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#ifndef REPS
#define REPS 100000
#endif

__global__ void empty()
{
    // do nothing!
}

int main(int argc, char *argv[])
{
    int ncores;
    int ndev;
    double *latency = NULL;
    double *latency_pp = NULL;
    double local_min_latency = DBL_MAX;
    double global_min_latency = DBL_MAX;
    const double usec = 1000.0 * 1000.0;

    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Determine number of cores and devices.
    cudaGetDeviceCount(&ndev);
    ncores = omp_get_num_procs();

    // get representative data to fill device
    int max_threads_per_block = 0;
    int max_threads_per_mp = 0;
    int mp_count = 0;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&max_threads_per_mp, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, 0);
    int n_blocks_to_start = (max_threads_per_mp / max_threads_per_block) * mp_count;

    // Allocate the memory to store the result data.
    latency_pp = (double *)malloc(ndev * sizeof(double));

    print_separator(rank);

    if (rank == 0)
    {
        fprintf(stderr, "number of cores for process: %d\n", ncores);
        fprintf(stderr, "number of processes: %d\n", world_size);
        fprintf(stderr, "number of devices: %d\n", ndev);
        fprintf(stderr, "number of repetitions: %d\n", REPS);
        fprintf(stderr, "mp_count: %d\n", mp_count);
        fprintf(stderr, "max_threads_per_block: %d\n", max_threads_per_block);
        fprintf(stderr, "max_threads_per_mp: %d\n", max_threads_per_mp);
        fprintf(stderr, "n_blocks_to_start: %d\n", n_blocks_to_start);
    }

    print_separator(rank);
    print_cpu_affinity(world_size, rank);
    print_separator(rank);

    // Perform some warm-up to make sure that all threads are up and running,
    // and the GPUs have been properly initialized.
    if (rank == 0)
    {
        fprintf(stderr, "warm up...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int c = 0; c < world_size; c++)
    {
        if (rank == c)
        {
            for (int d = 0; d < ndev; d++)
            {
                cudaSetDevice(d);
                empty<<<n_blocks_to_start, max_threads_per_block>>>();
                cudaDeviceSynchronize();
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    print_separator(rank);

    // Perform the actual measurements.
    if (rank == 0)
    {
        fprintf(stderr, "measurements...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int c = 0; c < world_size; c++)
    {
        if (rank == c)
        {
            for (int d = 0; d < ndev; d++)
            {
                fprintf(stderr, "running for process=%3d and device=%2d --> ", c, d);
                cudaSetDevice(d);

                double ts = omp_get_wtime();
                for (int r = 0; r < REPS; r++)
                {
                    empty<<<n_blocks_to_start, max_threads_per_block>>>();
                    cudaDeviceSynchronize();
                }
                double te = omp_get_wtime();
                latency_pp[d] = (te - ts) / ((double)REPS) * usec;
                if (latency_pp[d] < local_min_latency)
                {
                    local_min_latency = latency_pp[d];
                }
                fprintf(stderr, "avg. lat = %f\n", latency_pp[d]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    print_separator(rank);

    if (rank == 0)
    {
        latency = (double *)malloc(world_size * ndev * sizeof(double));
    }

    MPI_Gather(latency_pp, ndev, MPI_DOUBLE, latency, ndev, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_latency, &global_min_latency, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        print_results(world_size, ndev, latency, global_min_latency, 1);
        free(latency);
    }

    free(latency_pp);

    MPI_Finalize();

    return 0;
}