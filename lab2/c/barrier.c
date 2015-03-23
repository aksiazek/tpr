#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void computations() {
        double x = 123456789;
        for (int i = 0; i < 100; i++)
	        x = tan(exp(log(x)));
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size < 2) {
		fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1); 
	}

	int iterations = 10000;

	double start = MPI_Wtime();
	for (int i = 0; i < iterations; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		computations();
	}
	double end = MPI_Wtime();
	double elapsed_barrier = end - start;

	start = MPI_Wtime();
	for (int i = 0; i < iterations; i++) {
		computations();
	}
	end = MPI_Wtime();
	double elapsed_no_barrier = end - start;
	
	if (world_rank == 0)
		fprintf(stderr, "Barrier: %lf [s], no barrier: %lf [s]\n", elapsed_barrier, elapsed_no_barrier);

	MPI_Finalize();
	return 0;
}

