#include <stdio.h>
#include <omp.h>
#include <string.h>

#ifndef REPS
#define REPS 10
#endif

int main(int argc, char const * argv[]) {
    int ncores;
    int ndev;
    double *** bandwidth = NULL;
    const double usec = 1000.0 * 1000.0;

    const int nsizes = 7;
    size_t array_sizes_bytes[7] = {1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

    // Determine number of cores and devices.
    ndev = omp_get_num_devices();
    ncores = omp_get_num_procs();

    fprintf(stdout, "---------------------------------------------------------------\n");
    fprintf(stdout, "number of array sizes: %d\n", nsizes);
    fprintf(stdout, "number of cores:   %d\n", ncores);
    fprintf(stdout, "number of devices: %d\n", ndev);    
    fprintf(stdout, "number of repetitions: %d\n", REPS);
    fprintf(stdout, "---------------------------------------------------------------\n");

    // Allocate the memory to store the result data.
    bandwidth = (double ***)malloc(nsizes * sizeof(double **));
    for (int s = 0; s < nsizes; s++) {
        bandwidth[s] = (double **)malloc(ncores * sizeof(double *));
        for (int c = 0; c < ncores; c++) {
            bandwidth[s][c] = (double *)malloc(ndev * sizeof(double));
        }
    }

    // Print the OpenMP thread affinity info.
    #pragma omp parallel num_threads(ncores)
    {
        omp_display_affinity(NULL);
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
                    #pragma omp target device(d)
                    {
                        // do nothing
                    }
                    if (!d) {
                        fprintf(stdout, "#");
                        fflush(stdout);
                    }
                    else {
                        fprintf(stdout, ".");
                        fflush(stdout);
                    }
                }
            }
            #pragma omp barrier
        }
    }
    fprintf(stdout, "\n");
    fprintf(stdout, "---------------------------------------------------------------\n");

    // Perform the actual measurements.
    fprintf(stdout, "measurements...\n");
    #pragma omp parallel num_threads(ncores)
    {
        for (int c = 0; c < ncores; c++) {
            if (omp_get_thread_num() == c) {
                for (int s = 0; s < nsizes; s++) {
                    size_t cur_size     = array_sizes_bytes[s];
                    double tmp_size_mb  = ((double)cur_size / 1e6);

                    for (int d = 0; d < ndev; d++) {
                        // allocate and initialize data once (first-touch)
                        char * buffer = (char *)malloc(cur_size);
                        memset(buffer, 0, cur_size);

                        double ts = omp_get_wtime();
                        for (int r = 0; r < REPS; r++) {
                            #pragma omp target device(d) map(tofrom:buffer[0:cur_size])
                            {
                                // only touch single element
                                buffer[0] = 1;
                            }
                        }
                        double te = omp_get_wtime();
                        double avg_time_sec = (te - ts) / ((double) REPS);
                        bandwidth[s][c][d] = tmp_size_mb / avg_time_sec;
                        if (!d) {
                            fprintf(stdout, "#");
                            fflush(stdout);
                        }
                        else {
                            fprintf(stdout, ".");
                            fflush(stdout);
                        }

                        // free memory again
                        free(buffer);
                    }
                }
            }
            #pragma omp barrier
        }
    }
    fprintf(stdout, "\n");
    fprintf(stdout, "---------------------------------------------------------------\n");

    // Output the result data as CSV to the console.
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

    // Output the result data as CSV to the console.
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

    return 0;
}