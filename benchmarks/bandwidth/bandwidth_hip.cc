#include "../common/util.h"
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <omp.h>

#ifndef REPS
#define REPS 10
#endif

#ifndef INCLUDE_ALLOC
#define INCLUDE_ALLOC 0
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

__global__ void empty(size_t n, char *array)
{
    // do nothing!
}

int main(int argc, char *argv[])
{
    // ######################################################################
    // ### Variable declaration
    // ######################################################################

    const int nsizes = 3;
    size_t array_sizes_bytes[3] = {10000000, 100000000, 1000000000};
    const size_t MAX_BUF_SIZE = 1000000000;
    int ncores, ndev;
    double *times_abs = NULL;     // local measurements for absolute time
    double *bandwidth = NULL;     // local measurements for bandwidth
    double *min_bandwidth = NULL; // minimum bandwidth per size

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
        fprintf(stderr, "number of array sizes: %d\n", nsizes);
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

    times_abs = (double *)malloc(nsizes * ndev * sizeof(double));
    bandwidth = (double *)malloc(nsizes * ndev * sizeof(double));
    min_bandwidth = (double *)malloc(nsizes * sizeof(double));
    for (int s = 0; s < nsizes; s++)
    {
        min_bandwidth[s] = DBL_MAX;
    }

    char *buffer = (char *)malloc(MAX_BUF_SIZE); // buffer for transfer
    memset(buffer, 0, MAX_BUF_SIZE);

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
                empty<<<KERNEL_N, 64, 0, stream[d]>>>(d, nullptr);
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
            for (int s = 0; s < nsizes; s++)
            {
                size_t cur_size = array_sizes_bytes[s];
                double tmp_size_mb = ((double)cur_size / 1e6);
                for (int d = 0; d < ndev; d++)
                {
                    fprintf(stderr, "running for process=%3d, size=%7.2fMB and device=%2d --> ", c, tmp_size_mb, d);
                    HIPCALL(hipSetDevice(d));

                    char *buffer_dev = nullptr;
#if !INCLUDE_ALLOC
                    HIPCALL(hipMalloc(&buffer_dev, sizeof(*buffer_dev) * cur_size));
#endif

                    double ts = omp_get_wtime();
                    for (int r = 0; r < REPS; r++)
                    {
#if INCLUDE_ALLOC
                        HIPCALL(hipMalloc(&buffer_dev, sizeof(*buffer_dev) * cur_size));
#endif
                        HIPCALL(hipMemcpyHtoDAsync(buffer_dev, buffer, sizeof(*buffer) * cur_size, stream[d]));
                        empty<<<KERNEL_N, 64, 0, stream[d]>>>(cur_size, buffer_dev);
                        HIPCALL(hipMemcpyDtoHAsync(buffer, buffer_dev, sizeof(*buffer) * cur_size, stream[d]));
#if INCLUDE_ALLOC
                        HIPCALL(hipStreamSynchronize(stream[d]));
                        HIPCALL(hipFree(buffer_dev));
#endif
                    }
#if !INCLUDE_ALLOC
                    HIPCALL(hipStreamSynchronize(stream[d]));
#endif
                    double te = omp_get_wtime();
                    double avg_time_sec = (te - ts) / ((double)REPS);
                    times_abs[s * ndev + d] = avg_time_sec;
                    bandwidth[s * ndev + d] = tmp_size_mb * 2 / avg_time_sec;
                    if (bandwidth[s * ndev + d] < min_bandwidth[s])
                    {
                        min_bandwidth[s] = bandwidth[s * ndev + d];
                    }
                    fprintf(stderr, "avg. bw = %f\n", bandwidth[s * ndev + d]);
#if !INCLUDE_ALLOC
                    HIPCALL(hipFree(buffer_dev));
#endif
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    print_separator(rank);

    // ######################################################################
    // ### Gathering and printing results
    // ######################################################################

    double *global_times_abs = NULL;
    double *global_bandwidth = NULL;
    double *global_min_bandwidth = NULL;

    if (rank == 0)
    {
        global_times_abs = (double *)malloc(world_size * nsizes * ndev * sizeof(double));
        global_bandwidth = (double *)malloc(world_size * nsizes * ndev * sizeof(double));
        global_min_bandwidth = (double *)malloc(nsizes * sizeof(double));
    }

    MPI_Gather(times_abs, nsizes * ndev, MPI_DOUBLE, global_times_abs, nsizes * ndev, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(bandwidth, nsizes * ndev, MPI_DOUBLE, global_bandwidth, nsizes * ndev, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Reduce(min_bandwidth, global_min_bandwidth, nsizes, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        print_bw_results(world_size, nsizes, ndev, array_sizes_bytes, global_times_abs, global_bandwidth,
                         global_min_bandwidth);
    }

    // ######################################################################
    // ### Cleanup and finalization
    // ######################################################################

    if (rank == 0)
    {
        free(global_times_abs);
        free(global_bandwidth);
        free(global_min_bandwidth);
    }
    free(buffer);
    free(bandwidth);
    free(times_abs);
    free(min_bandwidth);
    delete[] stream;

    MPI_Finalize();

    return 0;
}