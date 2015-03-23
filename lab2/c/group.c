#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void run(int world_rank, int world_size, int test_num) {
        int buff_len = 1000; // buffered
	int buffer[buff_len];
	int repeats = 2000;
	buffer[0] = 1;

	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
        int* recv_buf = NULL; 
        int sum = 0;

	if (world_rank == 0)
		fprintf(stdout, "world_size = %i, test_num = %i\n", world_size, test_num);

	for (int i = 0; i < repeats; i++) {
		switch (test_num) {
		        case 0:
				if (world_rank == 0) {
					for (int whom = 1; whom < world_size; whom++)
						MPI_Send(buffer, buff_len, MPI_INT, whom, 0, MPI_COMM_WORLD);
				} else {
					MPI_Recv(buffer, buff_len, MPI_INT, 0, 0, 
					        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				break;
		
			case 1:
				MPI_Bcast(buffer, buff_len, MPI_INT, 0, MPI_COMM_WORLD);
				break;

			case 2:
			        if (world_rank == 0) {
					recv_buf = (int*)malloc(world_size * buff_len * sizeof(int));
				        sum = buffer[0];
				        for (int rank = 1; rank < world_size; rank++) {
				                MPI_Recv(recv_buf, world_size * buff_len, MPI_INT, rank, 0, 
				                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				                sum += recv_buf[0]; 
                                        }
				} else {
				        MPI_Send(buffer, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				}
				
				if (world_rank == 0) {
				        free(recv_buf);
				}
					
				break;
                        case 3:
				if (world_rank == 0)
					recv_buf = (int*)malloc(world_size * buff_len * sizeof(int));
				MPI_Reduce(buffer, recv_buf, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
				if (world_rank == 0)
					free(recv_buf);
				break;
			
		}
	}
		
	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();
        int total_size = 0;
        
        if (test_num < 2)
	        total_size += sizeof(int) * buff_len * repeats;
	else
	        total_size += sizeof(int) * repeats;
	double total_time = end - start;

	if (world_rank == 0) {
		printf("%lf [B/s]\n", total_size / total_time);
	}
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
	
	for (int mode = 0; mode < 4; mode++)
	        run(world_rank, world_size, mode);
        
	MPI_Finalize();
	return 0;
}

