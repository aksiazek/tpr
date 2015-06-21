/*** Calculating a derivative with CD ***/
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#define BSZ (16)

void checkErrors(char *label)
{
// we need to synchronise first to catch errors due to
// asynchroneous operations that would otherwise
// potentially go unnoticed
        cudaError_t err;
        err = cudaThreadSynchronize();
        if (err != cudaSuccess)
        {
                char *e = (char*) cudaGetErrorString(err);
                fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
                char *e = (char*) cudaGetErrorString(err);
                fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
        }
}

double get_time() 
{  
  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

double get_time_cpu() 
{  
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

texture<float, 2> tex_u;
texture<float, 2> tex_u_prev;

// GPU kernels
__global__ void copy_kernel (float *u, float *u_prev, int N, int BLOCKSIZE, int N_max)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int x = i + blockIdx.x*BLOCKSIZE;
	int y = j + blockIdx.y*BLOCKSIZE;
	int I = x + y*N_max;
	

	//if (I>=N*N){return;}	
	//if ((x>=N) || (y>=N)){return;}	
	float value = tex2D(tex_u, x, y);

	u_prev[I] = value;

}

__global__ void update_texture (float *u, float *u_prev, int N, float h, float dt, float alpha, int BLOCKSIZE, int N_max)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int x = i + blockIdx.x*BLOCKSIZE;
	int y = j + blockIdx.y*BLOCKSIZE;
	int I = x + y*N_max;
	
	//if (I>=N*N){return;}	
	//if ((x>=N) || (y>=N)){return;}	
	

	float t, b, r, l, c;
	c = tex2D(tex_u_prev, x, y);	
	t = tex2D(tex_u_prev, x, y+1);	
	b = tex2D(tex_u_prev, x, y-1);	
	r = tex2D(tex_u_prev, x+1, y);	
	l = tex2D(tex_u_prev, x-1, y);


	//if ( (I>N) && (I< N*N-1-N) && (I%N!=0) && (I%N!=N-1))
	if ( (x!=0) && (y!=0) && (x!=N-1) && (y!=N-1))
	{	
	        u[I] = c + alpha*dt/h/h * (t + b + l + r - 4*c);	
	}
}

void update_cpu(int x, int y, float *u, float *u_prev, int N, float h, float dt, float alpha)
{
	int I = N*y + x;
	u_prev[I] = u[I];

        // if not boundary do
        if ( (I > N) && (I < N*N-1-N) && (I % N != 0) && (I % N != N-1)) 
        {	
        u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
        }	

}


// GPU kernel
__global__ void copy_array(float *u, float *u_prev, int N)
{
        int i = threadIdx.x;
        int j = threadIdx.y;
        int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
        if (I>=N*N){return;}    
        u_prev[I] = u[I];

}

// GPU kernel
__global__ void update_global (float *u, float *u_prev, int N, float h, float dt, float alpha)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
	
	if (I>=N*N){return;}	
	//if (()>=N || j>){return;}	

	
	// if not boundary do
	if ( (I>N) && (I< N*N-1-N) && (I%N!=0) && (I%N!=N-1)) 
	{	u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
	}
	
	// Boundary conditions are automatically imposed
	// as we don't touch boundaries
}


__global__ void update_shared (float *u, float *u_prev, int N, float h, float dt, float alpha)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
	
	if (I>=N*N){return;}	

	__shared__ float u_prev_sh[BSZ][BSZ];

	u_prev_sh[i][j] = u_prev[I];
	
	__syncthreads();
	
	bool bound_check = ((I>N) && (I< N*N-1-N) && (I%N!=0) && (I%N!=N-1)); 
	bool block_check = ((i>0) && (i<BSZ-1) && (j>0) && (j<BSZ-1));

	// if not on block boundary do 
	if (block_check)
	{	u[I] = u_prev_sh[i][j] + alpha*dt/h/h * (u_prev_sh[i+1][j] + u_prev_sh[i-1][j] + u_prev_sh[i][j+1] + u_prev_sh[i][j-1] - 4*u_prev_sh[i][j]);
	}
	// if not on boundary 
	else if (bound_check) 
	//if (bound_check) 
	{	u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
	}
	
	// Boundary conditions are automatically imposed
	// as we don't touch boundaries
}

int main(int argc, char* argv[])
{
        
	// Allocate in CPU
	int N = 128;		// For textures to work, N needs to be a multiple of
	int BLOCKSIZE = 16;	// 32. As I will be using BLOCKSIZE to be a multiple of 8
						// I'll just look for the closest multiple of BLOCKSIZE (N_max)

        double elapsed, start, stop;
        int N_max = (int((N-0.5)/BLOCKSIZE) + 1) * BLOCKSIZE;

        if(argc > 1) {
                N = atoi(argv[1]);
        }
        
        if(argc == 1) {
                argv[2] = "global";
        }
        
        float xmin 	= 0.0f;
        float xmax 	= 3.5f;
        float h   	= (xmax-xmin)/(N-1);
        float dt	= 0.00001f;	
        float alpha	= 0.645f;
        float time 	= 0.4f;

        int steps = (int) ceil(time/dt);
        int I;
        
        if(strcmp(argv[2], "cpu") != 0) {
        
                if ((strcmp(argv[2], "texture") != 0))
	                cudaSetDevice(2);

	        float *x  	= new float[N*N]; 
	        float *y  	= new float[N*N]; 
	        float *u;
	
	        if ((strcmp(argv[2], "texture") != 0)) {
	                u = new float[N*N];
	        } else {
	                u = new float[N_max*N_max];
	        }
	                
	        float *u_prev  	= new float[N*N];

	        // Generate mesh and intial condition
	        if ((strcmp(argv[2], "texture") == 0)) {
	                // Initialize more bounds
	                for (int j=0; j<N_max; j++)
	                {	for (int i=0; i<N_max; i++)
		                {	I = N_max*j + i;
			                u[I] = 0.0f;
			                if ( ((i==0) || (j==0)) && (j<N) && (i<N)) 
				                {u[I] = 200.0f;}
		                }
	                }
	        } else {
	                // Generate mesh and intial condition
	                for (int j=0; j<N; j++)
	                {	for (int i=0; i<N; i++)
		                {	I = N*j + i;
			                u[I] = 0.0f;
			                if ( (i==0) || (j==0)) 
				                {u[I] = 200.0f;}
		                }
	                }	
	        }

	        // Allocate in GPU
	        float *u_d, *u_prev_d;

                if ((strcmp(argv[2], "texture") == 0)) {
                
                        cudaMalloc( (void**) &u_d, N_max*N_max*sizeof(float));
                        cudaMalloc( (void**) &u_prev_d, N_max*N_max*sizeof(float));
                
                        // Bind textures
	                cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	                cudaBindTexture2D(NULL, tex_u, u_d, desc, N_max, N_max, sizeof(float)*N_max);
	                cudaBindTexture2D(NULL, tex_u_prev, u_prev_d, desc, N_max, N_max, sizeof(float)*N_max);
	                
	                // Copy to GPU
	                cudaMemcpy(u_d, u, N_max*N_max*sizeof(float), cudaMemcpyHostToDevice);
	        } else {
	                
	                cudaMalloc( (void**) &u_d, N*N*sizeof(float));
	                cudaMalloc( (void**) &u_prev_d, N*N*sizeof(float));
	
	                // Copy to GPU
	                cudaMemcpy(u_d, u, N*N*sizeof(float), cudaMemcpyHostToDevice);
	
	        }

	

	        // Loop 
	        dim3 dimGrid(int((N-0.5)/BLOCKSIZE)+1, int((N-0.5)/BLOCKSIZE)+1);
	        dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	        start = get_time();
	
	        if((strcmp(argv[2], "global") == 0)) {
	                for (int t=0; t<steps; t++)
	                {	copy_array <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N);
	                        update_global <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha);
                               
                        }
		
	        } else if ((strcmp(argv[2], "shared") == 0)) {
	                for (int t=0; t<steps; t++)
	                {	copy_array <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N);
	                        update_shared <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha);
                               
                        }
	
	        } else if ((strcmp(argv[2], "texture") == 0)) {
	                for (int t=0; t<steps; t++)
	                {	
	                     copy_kernel <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, BLOCKSIZE, N_max);
		             update_texture <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE, N_max);
                               
                        }
	
	        } else {
	                std::cout<<"second argument should be global|shared|texture|cpu"<<std::endl;
	                exit(-1);
	        }
	
	        stop = get_time();

                /*char str[40];
                strcat(str, "update_");
                strcat(str, argv[2]);
                checkErrors(str);*/
                
	        elapsed = stop - start;
	
	        // Copy result back to host
	        // cudaMemcpy(u, u_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        
        
                // Free device
                if ((strcmp(argv[2], "texture") == 0)) {
                
                        cudaUnbindTexture(tex_u);
                        cudaUnbindTexture(tex_u_prev);
                }
	
	        cudaFree(u_d);
	        cudaFree(u_prev_d);
        
        
        } else {
                float *u_prev_cpu  = new float[N*N];
                float *u_cpu  	= new float[N*N];
        
                // Generate mesh and intial condition
	        for (int j=0; j<N; j++)
	        {	for (int i=0; i<N; i++)
		        {	I = N*j + i;
			        u_cpu[I] = 0.0f;
		                if ( (i==0) || (j==0)) 
			                {u_cpu[I] = 200.0f;}
		        }
	        }
        
                start = get_time();
                for (int t=0; t<steps; t++)
                {	
                     for (int j=0; j<N; j++)
                     {	
                        for (int i=0; i<N; i++)
	                {
	                        update_cpu (i, j, u_cpu, u_prev_cpu, N, h, dt, alpha);
	                }
	             }
                }
                stop = get_time_cpu();
                elapsed = stop - start;
        
        }

        std::cout << N << " " << elapsed << std::endl;
     
}
