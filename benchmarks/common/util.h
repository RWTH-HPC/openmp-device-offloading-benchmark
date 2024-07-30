#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <mpi.h>
#include <numa.h>
#include <sched.h>
#include <stdio.h>

void print_separator(int rank)
{
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "---------------------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_cpu_affinity(int world_size, int rank)
{
    for (int c = 0; c < world_size; c++)
    {
        if (rank == c)
        {
            int cpu = sched_getcpu();
            int numa = numa_node_of_cpu(cpu);
            fprintf(stderr, "Process %3d running at core %3d on NUMA domain %3d\n", rank, cpu, numa);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_results(int world_size, int ndev, double *data, double global_min_value, int is_latency)
{
    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, "Absolute measurements %s\n", is_latency ? "(us)" : "");
    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, ";");
    for (int c = 0; c < world_size; c++)
    {
        fprintf(stderr, "Process %d%c", c, c < world_size - 1 ? ';' : '\n');
    }

    for (int d = 0; d < ndev; d++)
    {
        fprintf(stderr, "GPU %d;", d);
        for (int c = 0; c < world_size; c++)
        {
            fprintf(stderr, "%lf%c", data[c * ndev + d], c < world_size - 1 ? ';' : '\n');
        }
    }

    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, "Relative measurements to minimum %s\n", is_latency ? "(us)" : "");
    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, ";");
    for (int c = 0; c < world_size; c++)
    {
        fprintf(stderr, "Process %d%c", c, c < world_size - 1 ? ';' : '\n');
    }
    for (int d = 0; d < ndev; d++)
    {
        fprintf(stderr, "GPU %d;", d);
        for (int c = 0; c < world_size; c++)
        {
            fprintf(stderr, "%lf%c", (data[c * ndev + d] / global_min_value), c < world_size - 1 ? ';' : '\n');
        }
    }
}