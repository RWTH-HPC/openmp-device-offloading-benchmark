#include <float.h>
#include <mpi.h>
#include <numa.h>
#include <omp.h>
#include <sched.h>
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
    double min_latency = DBL_MAX;
    double global_min_latency = DBL_MAX;
    const double usec = 1000.0 * 1000.0;

    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Determine number of cores and devices.
    ndev = omp_get_num_devices();
    ncores = omp_get_num_procs();

    if (rank == 0)
    {
        fprintf(stdout, "---------------------------------------------------------------\n");
        fprintf(stdout, "number of cores on system:   %d\n", ncores);
        fprintf(stdout, "number of processes:   %d\n", world_size);
        fprintf(stdout, "number of devices: %d\n", ndev);
        fprintf(stdout, "number of repetitions: %d\n", REPS);
        fprintf(stdout, "---------------------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate the memory to store the result data.
    latency_pp = (double *)malloc(ndev * sizeof(double *));

    for (int c = 0; c < world_size; c++)
    {
        if (rank == c)
        {
            int cpu = sched_getcpu();
            int numa = numa_node_of_cpu(cpu);
            fprintf(stdout, "Process %3d running at core %3d on NUMA domain %3d\n", rank, cpu, numa);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0)
        fprintf(stdout, "---------------------------------------------------------------\n");

    // Perform some warm-up to make sure that all threads are up and running,
    // and the GPUs have been properly initialized.
    if (rank == 0)
        fprintf(stdout, "warm up...\n");
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
    if (rank == 0)
        fprintf(stdout, "---------------------------------------------------------------\n");

    // Perform the actual measurements.
    if (rank == 0)
        fprintf(stdout, "measurements...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    double val = 0;
    for (int c = 0; c < world_size; c++)
    {
        if (rank == c)
        {
            for (int d = 0; d < ndev; d++)
            {
                fprintf(stdout, "running for process=%3d and device=%2d\n", c, d);
                fflush(stdout);

                double ts = omp_get_wtime();
                for (int r = 0; r < REPS; r++)
                {
// #pragma omp target device(d) map(tofrom:val)
#pragma omp target device(d)
                    {
                        // do nothing
                        // val += c*d+r; // <== might avoid compiler from optimizing out stuff
                    }
                }
                double te = omp_get_wtime();
                latency_pp[d] = (te - ts) / ((double)REPS) * usec;
                if (latency_pp[d] < min_latency)
                {
                    min_latency = latency_pp[d];
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (rank == 0)
        fprintf(stdout, "dummy=%f\n", val);
    if (rank == 0)
        fprintf(stdout, "---------------------------------------------------------------\n");

    if (rank == 0)
    {
        latency = (double *)malloc(world_size * ndev * sizeof(double *));
    }

    MPI_Gather(latency_pp, ndev, MPI_DOUBLE, latency, ndev, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Reduce(&min_latency, &global_min_latency, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        fprintf(stdout, "---------------------------------------------------------------\n");
        fprintf(stdout, "Absolute measurements (us)\n");
        fprintf(stdout, "---------------------------------------------------------------\n");
        fprintf(stdout, ";");
        for (int c = 0; c < world_size; c++)
        {
            fprintf(stdout, "Process %d%c", c, c < world_size - 1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++)
        {
            fprintf(stdout, "GPU %d;", d);
            for (int c = 0; c < world_size; c++)
            {
                fprintf(stdout, "%lf%c", latency[c * world_size + d], c < world_size - 1 ? ';' : '\n');
            }
        }

        fprintf(stdout, "---------------------------------------------------------------\n");
        fprintf(stdout, "Relative measurements to minimum latency\n");
        fprintf(stdout, "---------------------------------------------------------------\n");
        fprintf(stdout, ";");
        for (int c = 0; c < world_size; c++)
        {
            fprintf(stdout, "Process %d%c", c, c < world_size - 1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++)
        {
            fprintf(stdout, "GPU %d;", d);
            for (int c = 0; c < world_size; c++)
            {
                fprintf(stdout, "%lf%c", (latency[c * world_size + d] / global_min_latency),
                        c < world_size - 1 ? ';' : '\n');
            }
        }
        free(latency);
    }

    free(latency_pp);

    MPI_Finalize();

    return 0;
}