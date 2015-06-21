#include <iostream>
#include <cublas.h>
#include <time.h>

#define N 10

using namespace std;

int main(int argc, char* argv[]) {
        cublasInit();
        int version;
        cublasGetVersion(&version);
        cout << "Cublas version: " << version << endl;
        
        srand(time(NULL));
        
        double* array1  = new double[N];
        double* array2  = new double[N];
        double** matrix1  = new double*[N];
        double** matrix2  = new double*[N];
        double* v1_d;
        double* v2_d;
        double* m1_d;
        double* m2_d;
        
        for(int i = 0; i < N; i++) {
	        array1[i] = 1;
                array2[i] = 1;
                matrix1[i]  = new double[N];
                matrix2[i]  = new double[N];
        }
        
        for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                        matrix1[i][j] = 4;
                        matrix2[i][j] = (rand()+0.0) / INT_MAX;
                }
        }
        
        cudaError_t err;
        cudaMalloc((void**) &v1_d, N*sizeof(double));
        cudaMalloc((void**) &v2_d, N*sizeof(double));
        cudaMalloc((void**) &m1_d, N*N*sizeof(double));
        cudaMalloc((void**) &m1_d, N*N*sizeof(double));
        
        if(cublasSetVector(N, sizeof(double), array1, 1, v1_d, 1) != CUBLAS_STATUS_SUCCESS)
                cout << "Error in set vector" << endl;
        if(cublasSetVector(N, sizeof(double), array2, 1, v2_d, 1) != CUBLAS_STATUS_SUCCESS)
                cout << "Error in set vector" << endl;
        
        if(cublasSetMatrix(N, N, sizeof(double), matrix1, 1, m1_d, 1) != CUBLAS_STATUS_SUCCESS)
                cout << "Error in set matrix" << endl;
        if(cublasSetMatrix(N, N, sizeof(double), matrix2, 1, m2_d, 1) != CUBLAS_STATUS_SUCCESS)
                cout << "Error in set matrix" << endl;
        cublasDaxpy(N, 1.0, v1_d, 1, v2_d, 1);
        cublasDgemv('n', N, N, 1.0, m1_d, 1, v2_d, 1, 0.0, v2_d, 1);
          
        if(cublasGetVector(N, sizeof(double), v2_d, 1, array2, 1) != CUBLAS_STATUS_SUCCESS)
                cout << "Error in get vector" << endl;
              
        cublasGetMatrix(N, N, sizeof(double), m2_d, 1, matrix2, 1);      
              
        for(int i = 0; i < N; i++) {
	        cout << array2[i] << endl;
        }
        
        cout << "now matrix" << endl << endl << endl << endl;
        
        for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                        cout << matrix2[i][j] << endl;
                }
	        
        }
        
        cudaFree(v1_d);
        cudaFree(v2_d);
        cudaFree(m1_d);
        cudaFree(m2_d);
        delete[] array1;
        delete[] array2;
        for(int i = 0; i < N; i++) {
	        delete[] matrix1[i];
                delete[] matrix2[i];
        }
        delete[] matrix1;
        delete[] matrix2;
        cublasShutdown();
        return 0;
}
