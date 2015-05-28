#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "helper_timer.h"

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__ )

inline void __cudaCheckError(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s at (%s:%i)\n",
                 cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

__global__ void add (int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a[tid]+b[tid];
    }
}

void add_cpu(int *a, int *b, int *c, int N, unsigned Blocks, unsigned Threads, FILE * pFile, float time) {
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    
    fprintf(pFile, "%u %u %u %f %ld.%06ld\n", Blocks, Threads, N, time, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}

void check (int* cpu_c, int* gpu_c, int N) {

    int flag = 0;
    for (int i = 0; i < N; i++) {
        if(gpu_c[i] != cpu_c[i]) {
            printf("Not Equal!\n");
            flag = 1;
            break;
        }
    }
    if(!flag) {
        printf("Equal!\n");
    }
}

int main(int argc, char* argv[]) {

    FILE * gFile = fopen("results", "a");

    if(argc != 4) {
        printf("Usage: %s [liczba-blocków] [wątki-na-block] [rozmiar-tablicy]\n", argv[0]);
        exit(-1);
    }

    unsigned Blocks = atoi(argv[1]);
    unsigned Threads = atoi(argv[2]);
    unsigned N = atoi(argv[3]);

    int* a = (int*) malloc(N * sizeof(int));
    int* b = (int*) malloc(N * sizeof(int));
    int* c = (int*) malloc(N * sizeof(int));
    int* cpu_a = (int*) malloc(N * sizeof(int));
    int* cpu_b = (int*) malloc(N * sizeof(int));
    int* cpu_c = (int*) malloc(N * sizeof(int));
    int *dev_a, *dev_b, *dev_c;//, *a_d;
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
        cpu_a[i] = i;
        cpu_b[i] = i*2;
    }
    cudaCheckError(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice));
    
    StopWatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    
    add <<<Blocks,Threads>>> (dev_a,dev_b,dev_c, N);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaThreadSynchronize());
    sdkStopTimer(&timer);
    float time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    cudaCheckError(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));
    /*for (int i = 0; i < N; i++) {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }*/
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    add_cpu(cpu_a, cpu_b, cpu_c, N, Blocks, Threads, gFile, time/1000);
    //check(cpu_c, c, N);
    /*for (int i = 0; i < N; i++) {
        printf("%d+%d=%d\n", cpu_a[i], cpu_b[i], cpu_c[i]);
    }*/
    
    fclose (gFile);
    free(a);
    free(b);
    free(c);
    free(cpu_a);
    free(cpu_b);
    free(cpu_c);
    
    return 0;
}
