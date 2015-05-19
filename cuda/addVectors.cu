#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "helper_timer.h"

__global__ void add (int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a[tid]+b[tid];
    }
}

void add_cpu(int *a, int *b, int *c, int N) {
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    
    printf("Time or the CPU: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}

void check (int* cpu_c, int* gpu_c, int N) {

    int flag = 0;
    for (int i = 0; i < N; i++) {
        if(gpu_c[i] != cpu_c[i]) {
            printf("Not Equal!");
            flag = 1;
            break;
        }
    }
    if(!flag) {
        printf("Equal!");
    }
}

int main(int argc, char* argv[]) {

    if(argc != 3) {
        printf("Usage: %s [liczba-blocków] [wątki-na-block]", argv[0]);
        exit(-1);
    }

    unsigned B = atoi(argv[1]);
    int N = atoi(argv[2]);

    int a[N],b[N],c[N];
    int cpu_a[N], cpu_b[N], cpu_c[N];
    int *dev_a, *dev_b, *dev_c;//, *a_d;
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);
    
    StopWatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    
    add <<<B,N>>> (dev_a,dev_b,dev_c, N);
    
    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    for (int i = 0; i < N; i++) {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }
    
    printf ("Time for the kernel: %f ms\n", time);
    
    add_cpu(cpu_a, cpu_b, cpu_c, N);
    for (int i = 0; i < N; i++) {
        printf("%d+%d=%d\n", cpu_a[i], cpu_b[i], cpu_c[i]);
    }
    
    check(cpu_c, c, N);
    
    return 0;
}
