#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFFSIZE 9000000
#define BUFFERED_MODE 1
#define SYNCHRONISED_MODE 2
#define BITS_IN_MBIT 1000000
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    char buffer[BUFFSIZE];
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int numOfCom=200;
    int maxMsgSize=262144;
    int mode=SYNCHRONISED_MODE;

    double start=0;
    double end=0;

    int i=0,j=0;

    if ( argc != 4 )
    {
        printf( "usage: %s times_to_send max_message_size communication_mode\nUsing defaults: %d %d %d\n", argv[0],numOfCom, maxMsgSize, mode );
    } else {
        numOfCom = atoi(argv[1]);
        maxMsgSize = atoi(argv[2]);
        mode = atoi(argv[3]);
    }

    char msgBuf[maxMsgSize];

    int buffsize;
    if(mode==BUFFERED_MODE) {
        buffsize = BUFFSIZE;
        MPI_Buffer_attach(buffer,BUFFSIZE);
    }

    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for(i=0; i<maxMsgSize; i++) {
        msgBuf[i]=i;
    }
    if (world_rank == 0) {
        for(j=1; j<maxMsgSize; j*=2) {
            MPI_Barrier(MPI_COMM_WORLD);
            for(i=0; i<numOfCom; i++) {
                if(mode==BUFFERED_MODE) {
                    MPI_Bsend(msgBuf, j, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                } else {
                    MPI_Ssend(msgBuf, j, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else if (world_rank == 1) {
        for(j=1; j<maxMsgSize; j*=2) {
            MPI_Barrier(MPI_COMM_WORLD);
            start = MPI_Wtime();
            for(i=0; i<numOfCom; i++) {
                MPI_Recv(msgBuf, j, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            end = MPI_Wtime();
            double time = end-start;
            printf("%f\n", (j*8*numOfCom)/ time / BITS_IN_MBIT);
        }
    }

    if(mode==BUFFERED_MODE) {
        MPI_Buffer_detach(buffer, &buffsize);
    }

    MPI_Finalize();
}

