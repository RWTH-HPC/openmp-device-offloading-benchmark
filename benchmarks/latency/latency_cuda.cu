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
    // ######################################################################
    // ### Variable declaration
    // ######################################################################

    int ncores, ndev;
    double *latency = NULL;
    double local_min_latency = DBL_MAX;
    double global_min_latency = DBL_MAX;
    const double usec = 1000.0 * 1000.0;

    // ######################################################################
    // ### MPI initialization + additional info
    // ######################################################################

    int rank, world_size;
    MPI_Init(&argc, &argv);
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

    // ######################################################################
    // ### Memory allocation
    // ######################################################################

    // Allocate the memory to store the result data.
    latency = (double *)malloc(ndev * sizeof(double));

    // ######################################################################
    // ### Perform some warm-up to make sure that all threads are up and
    // ### running, and the GPUs have been properly initialized.
    // ######################################################################

    print_from_root(rank, "warm up...\n");

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

    // ######################################################################
    // ### Perform the actual measurements
    // ######################################################################

    print_from_root(rank, "measurements...\n");

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
                latency[d] = (te - ts) / ((double)REPS) * usec;
                if (latency[d] < local_min_latency)
                {
                    local_min_latency = latency[d];
                }
                fprintf(stderr, "avg. lat = %f\n", latency[d]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    print_separator(rank);

    // ######################################################################
    // ### Gathering and printing results
    // ######################################################################

    double *global_latency = NULL;
    if (rank == 0)
    {
        global_latency = (double *)malloc(world_size * ndev * sizeof(double));
    }

    MPI_Gather(latency, ndev, MPI_DOUBLE, global_latency, ndev, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_latency, &global_min_latency, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        print_lat_results(world_size, ndev, global_latency, global_min_latency);
    }

    // ######################################################################
    // ### Cleanup and finalization
    // ######################################################################
    if (rank == 0)
    {
        free(global_latency);
    }
    free(latency);

    MPI_Finalize();

    return 0;
}