#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFFSIZE 2000
#define BUFFERED_MODE 1
#define SYNCHRONISED_MODE 2
int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  char buffer[BUFFSIZE];
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int numOfCom=2000;
  int mode=SYNCHRONISED_MODE;

  double start=0;
  double end=0;

  int i=0;
  
  if ( argc != 3 )
  {
    printf( "usage: %s times_to_send communication_mode\nUsing defaults: %d %d\n", argv[0],numOfCom, mode );
  } else {
    numOfCom = atoi(argv[1]);
	mode = atoi(argv[2]);
  }
  
  if(mode==BUFFERED_MODE){
      MPI_Buffer_attach(buffer,BUFFSIZE);
  }

  if (world_size < 2) {
    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1); 
  }
  char message;
  if (world_rank == 0) {
    message = 'a';
    MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
    for(i=0;i<numOfCom;i++){
	    if(mode==BUFFERED_MODE){
          MPI_Bsend(&message, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
	    } else {
		  MPI_Ssend(&message, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		}
		MPI_Recv(&message, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
	end = MPI_Wtime();
    printf("Process 1 received message %c from process 0\n", message);
    printf("Time elapsed: %f\n", (end - start)*1000/(numOfCom*2));
  } else if (world_rank == 1) {
    MPI_Barrier(MPI_COMM_WORLD);

    for(i=0;i<numOfCom;i++){
      MPI_Recv(&message, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  if(mode==BUFFERED_MODE){
          MPI_Bsend(&message, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	    } else {
		  MPI_Ssend(&message, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
       }
    }
  }
  MPI_Finalize();
}
