#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <mpi.h>
#include <numa.h>
#include <sched.h>
#include <stdio.h>

void print_from_root(int rank, const char *msg)
{
    if (rank == 0)
    {
        fprintf(stderr, "%s", msg);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

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

void print_lat_results(int world_size, int ndev, double *data, double global_min_value)
{
    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, "Absolute measurements (us)\n");
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
    fprintf(stderr, "Relative measurements to minimum (us)\n");
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

void print_bw_results(int world_size, int nsizes, int ndev, size_t *array_sizes_bytes, double *data_time,
                      double *data_bw, double *global_min_value)
{
    int stride = nsizes * ndev;
    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, "Absolute times (sec)\n");
    fprintf(stderr, "---------------------------------------------------------------\n");
    for (int s = 0; s < nsizes; s++)
    {
        size_t cur_size = array_sizes_bytes[s];
        fprintf(stderr, "##### Problem Size: %.2f KB\n", cur_size / 1000.0);
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
                fprintf(stderr, "%lf%c", data_time[c * stride + s * ndev + d], c < world_size - 1 ? ';' : '\n');
            }
        }
    }

    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, "Absolute measurements (MB/s)\n");
    fprintf(stderr, "---------------------------------------------------------------\n");
    for (int s = 0; s < nsizes; s++)
    {
        size_t cur_size = array_sizes_bytes[s];
        fprintf(stderr, "##### Problem Size: %.2f KB\n", cur_size / 1000.0);
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
                fprintf(stderr, "%lf%c", data_bw[c * stride + s * ndev + d], c < world_size - 1 ? ';' : '\n');
            }
        }
    }
    fprintf(stderr, "\n\n");
    for (int c = 0; c < world_size; c++)
    {
        fprintf(stderr, "##### Process: %d\n", c);
        fprintf(stderr, ";");
        for (int s = 0; s < nsizes; s++)
        {
            size_t cur_size = array_sizes_bytes[s];
            fprintf(stderr, "%.2f KB%c", cur_size / 1000.0, s < nsizes - 1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++)
        {
            fprintf(stderr, "GPU %d;", d);
            for (int s = 0; s < nsizes; s++)
            {
                fprintf(stderr, "%lf%c", data_bw[c * stride + s * ndev + d], s < nsizes - 1 ? ';' : '\n');
            }
        }
    }

    fprintf(stderr, "---------------------------------------------------------------\n");
    fprintf(stderr, "Relative measurements to minimum bandwidth for size\n");
    fprintf(stderr, "---------------------------------------------------------------\n");
    for (int s = 0; s < nsizes; s++)
    {
        size_t cur_size = array_sizes_bytes[s];
        fprintf(stderr, "##### Problem Size: %.2f KB\n", cur_size / 1000.0);
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
                fprintf(stderr, "%lf%c", data_bw[c * stride + s * ndev + d] / global_min_value[s],
                        c < world_size - 1 ? ';' : '\n');
            }
        }
    }
    fprintf(stderr, "\n\n");
    for (int c = 0; c < world_size; c++)
    {
        fprintf(stderr, "##### Process: %d\n", c);
        fprintf(stderr, ";");
        for (int s = 0; s < nsizes; s++)
        {
            size_t cur_size = array_sizes_bytes[s];
            fprintf(stderr, "%.2f KB%c", cur_size / 1000.0, s < nsizes - 1 ? ';' : '\n');
        }
        for (int d = 0; d < ndev; d++)
        {
            fprintf(stderr, "GPU %d;", d);
            for (int s = 0; s < nsizes; s++)
            {
                fprintf(stderr, "%lf%c", data_bw[c * stride + s * ndev + d] / global_min_value[s],
                        s < nsizes - 1 ? ';' : '\n');
            }
        }
    }
}