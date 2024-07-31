#include "../common/util.h"
#include <cfloat>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <omp.h>

#ifndef REPS
#define REPS 100000
#endif

// least common multiple of 104 and 110 times wavefront size
#define KERNEL_N (5720 * 64)

// Define macro to automate error handling of the HIP API calls
#define HIPCALL(func)                                                                                                  \
    {                                                                                                                  \
        hipError_t ret = func;                                                                                         \
        if (ret != hipSuccess)                                                                                         \
        {                                                                                                              \
            fprintf(stderr, "HIP error: '%s' at %s:%d\n", hipGetErrorString(ret), __FUNCTION__, __LINE__);             \
            abort();                                                                                                   \
        }                                                                                                              \
    }

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
    HIPCALL(hipGetDeviceCount(&ndev));
    ncores = omp_get_num_procs();

    print_separator(rank);

    if (rank == 0)
    {
        fprintf(stderr, "number of cores for process: %d\n", ncores);
        fprintf(stderr, "number of processes: %d\n", world_size);
        fprintf(stderr, "number of devices: %d\n", ndev);
        fprintf(stderr, "number of repetitions: %d\n", REPS);
    }

    print_separator(rank);
    print_cpu_affinity(world_size, rank);
    print_separator(rank);

    // ######################################################################
    // ### Memory allocation
    // ######################################################################

    // Streams to the devices
    hipStream_t *stream = new hipStream_t[ndev];
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
                HIPCALL(hipSetDevice(d));
                HIPCALL(hipStreamCreate(&stream[d]));
                empty<<<KERNEL_N, 64, 0, stream[d]>>>();
                HIPCALL(hipStreamSynchronize(stream[d]));
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
                HIPCALL(hipSetDevice(d));

                double ts = omp_get_wtime();
                for (int r = 0; r < REPS; r++)
                {
                    empty<<<KERNEL_N, 64, 0, stream[d]>>>();
                }
                HIPCALL(hipStreamSynchronize(stream[d]));
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
    delete[] stream;

    MPI_Finalize();

    return 0;
}