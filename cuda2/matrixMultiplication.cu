// Matrix multiplication by parts
// Elements stored in row-major order

using namespace std;
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "helper_timer.h"
#define BLOCK_SIZE 16

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__ )

inline void __cudaCheckError(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s at (%s:%i)\n",
                 cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

typedef struct
{	int width;
	int height;
	float *elements;
} Matrix;

// Forward declaration of matrix mult
__global__ void MatMulKernel (const Matrix, const Matrix, Matrix);

void matrix_mul(const Matrix A, const Matrix B, Matrix C, float gpu_time)
{
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    
    for (int i = 0; i < A.height; i++)
	{	for (int j = 0; j < B.width; j++) 
	        {
	                C.elements[i*C.width+j] = 0;
	                for (int k = 0; k < A.width; k++) {
	                        C.elements[i*C.width+j] += A.elements[i*A.width+k] * B.elements[k*B.width+j];
	                }   
	        }
	}
	
	gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    
    printf("%u %u %f %ld.%06ld\n", A.width, A.height, gpu_time/1000, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}

void check(const Matrix GPU_C, const Matrix CPU_C)
{
    for (int i = 0; i < GPU_C.height; i++)
	{	
	    for (int j = 0; j < GPU_C.width; j++) 
        {
            if(GPU_C.elements[i*GPU_C.width+j] != CPU_C.elements[i*CPU_C.width+j]) 
            {
                printf("Not Equal!");
                return;
            }
                 
        }
			
	}

}

// Host code
float MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load matrices A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void**) &d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void**) &d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	
	// allocate C in device
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = d_C.width * d_C.height * sizeof(float);
	cudaMalloc((void**) &d_C.elements, size);
	
	// call kernel
    dim3 dimBlock(32, 32, 1); // threads per block?
    dim3 dimGrid(1024, 1024, 1); // number of blocks?
    
    StopWatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	cudaCheckError(cudaPeekAtLastError());
	cudaCheckError(cudaThreadSynchronize());
    sdkStopTimer(&timer);
    float time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
	
	// copy C to host
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	
	return time;
}

//matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{	
	// each thread computes one element of C and acumulates results to Cvalue
        float Cvalue = 0;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if ((row>=A.height)  || (col>=B.width))
                return;
           
        for (int e = 0; e < A.width; e++) {
                Cvalue += A.elements[row*A.width + e] * B.elements[e*B.width + col];
                C.elements[row*C.width + col] = Cvalue;
        }
        
}

int main(int argc, char * const argv[])
{	
    if(argc != 2)
        exit(-1);
        
	int Width = atoi(argv[1]);
	int Height = Width;
	
	Matrix A;
	Matrix B;
	Matrix C;
	Matrix Cpu_C;
	
	A.width = Width;
	B.width = Width;
	C.width = Width;
	Cpu_C.width = Width;
	
	A.height = Height;
	B.height = Height;
	C.height = Height;
	Cpu_C.height = Height;
	
	A.elements = new float[Width*Height];
	B.elements = new float[Width*Height];
	C.elements = new float[Width*Height];
	Cpu_C.elements = new float[Width*Height];
	
	//fill matrices
	std::ifstream A_input;
	std::ifstream B_input;
	A_input.open("A_256x256.txt");
	B_input.open("B_256x256.txt");
	
	float a, b;
	A_input >> a;	
	B_input >> b;	
	int i = 0;
	while (!A_input.eof())
	{	A.elements[i] = a;
		B.elements[i] = b;
		A_input >> a;	
		B_input >> b;	
		i += 1;
	}
	A_input.close();
	B_input.close();
	
	for(int j = i; j < A.width*A.height; j++) 
	{
	    A.elements[j] = A.elements[j % i];
	    B.elements[j] = B.elements[j % i];
	}

	float gpu_time = MatMul(A, B, C);
	printf("%u %u %f\n", A.width, A.height, gpu_time/1000);
	
	//matrix_mul(A, B, Cpu_C, gpu_time);
	//check(C, Cpu_C);
	
	/*std::ofstream C_output;
	C_output.open("C.txt");
	for (int i=0; i<Width; i++)
	{	for (int j=0; j<Width; j++)
	
			C_output<<C.elements[i*Width+j]<<"\t";
		C_output<<endl;
	}*/

}
	
