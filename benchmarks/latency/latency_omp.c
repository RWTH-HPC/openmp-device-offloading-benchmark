#include "../common/util.h"
#include <float.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#ifndef REPS
#define REPS 100000
#endif

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
    ndev = omp_get_num_devices();
    ncores = omp_get_num_procs();

    // Allocate the memory to store the result data.
    latency_pp = (double *)malloc(ndev * sizeof(double));

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
#pragma omp target device(d)
                {
                    // do nothing
                }
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

    double val = 0;
    for (int c = 0; c < world_size; c++)
    {
        if (rank == c)
        {
            for (int d = 0; d < ndev; d++)
            {
                fprintf(stderr, "running for process=%3d and device=%2d --> ", c, d);

                double ts = omp_get_wtime();
                for (int r = 0; r < REPS; r++)
                {
// #pragma omp target device(d) map(tofrom:val)
#pragma omp target device(d)
                    {
                        // do nothing
                        // val += c*d+r; // <== might avoid compiler from optimizing out codes or regions
                    }
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

    // cleanup
    free(latency_pp);
    MPI_Finalize();

    return 0;
}