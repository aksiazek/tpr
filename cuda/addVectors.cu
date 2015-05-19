#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "helper_timer.h"

__global__ void add (int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a[tid]+b[tid];
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
    // float a_h[N];
    int *dev_a, *dev_b, *dev_c;//, *a_d;
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    // cudaMalloc((void **) &a_d, sizeof(float)*N); // alokuj pamięć na GPU
    
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);
    
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
    for (int i = 0; i < N; i++) {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }
    
    // cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    printf ("Time for the kernel: %f ms\n", time);
    
    // cudaFree(a_d);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
