#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h> // for clock_gettime()

struct bucket {
	int count;
	double* values;
};

int compare(const void* first, const void* second)
{
    double a = *((double*)first), b =  *((double*)second);
    if (a < b)
    {
        return -1;
    }
    else if (a > b)
    {
        return 1;
    }
    else
    {
        return 0;
    }  
}

void bucketSort(double array[], int n, int bucket_count)
{
    int i, j, already;
    double left_range, right_range;
    double step = 1.0 / bucket_count;

    struct bucket* buckets = malloc(sizeof(struct bucket)*bucket_count);
    int* indexes = malloc(sizeof(int)*bucket_count);
    
    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < bucket_count; i++)
    {
        buckets[i].count = 0;
	buckets[i].values = malloc(sizeof(double)*n);
    }
   
    #pragma omp parallel for default(shared) private(i, j, left_range, right_range, already) schedule(dynamic)
    for (i = 0; i < n; i++)
    {
	left_range = 0;
	right_range = step;
	j = 0;
	already = 0;
	while(!already) {
		if((array[i] >= left_range) && (array[i] <= right_range)) {
			already = 1;
			
			#pragma omp critical
			buckets[j].values[buckets[j].count++] = array[i];
			
			//printf("add: %d, %d, %lf, %lf %lf\n", i, j, array[i], left_range, right_range);
		}
		j++;
		left_range+=step;
		right_range+=step;
	}
        
    }
     
    //for (i = 0; i < bucket_count; i++)
    //    printf("%d: %d\n", i, buckets[i].count);
    
    indexes[0] = 0;
    for (i = 1; i < bucket_count; i++)
    {
        indexes[i] = indexes[i-1] + buckets[i-1].count;
    }

    #pragma omp parallel for default(shared) private(i, j)
    for (i = 0; i < bucket_count; i++)
    {
        qsort(buckets[i].values, buckets[i].count, sizeof(double), &compare);
        for (j = 0; j < buckets[i].count; j++)
        { 
            //printf("%lf[%d] ", buckets[i].values[j], i);
            array[indexes[i] + j] = buckets[i].values[j];
        } 
	free(buckets[i].values);
    }
    free(buckets);
    free(indexes);
}

int main(int argc, char** argv)
{
        // omp_set_dynamic(0);     // Explicitly disable dynamic teams ?
        if(argc != 4) {
                printf("Usage: %s [how-much] [max-threads] [0|1 (meaning normal|scaled)]\n", argv[0]);
                exit(-1);
        }

        int n = atoi(argv[1]);
        int max_threads = atoi(argv[2]);
        int buckets;
        double* array;
        
        unsigned int randval;
        FILE* f = fopen("/dev/urandom", "r");
        fread(&randval, sizeof(randval), 1, f);
        fclose(f);
	srand(randval);
	
	double seq_time;
	double speedup, efficiency, karp_flatt;
        
        for(int threads = 1; threads <= max_threads; threads++) {
        
                if(atoi(argv[3]) == 1) {
                        n = atoi(argv[1]) * threads;
                }
        
                buckets = n / 1000 + 1;
                array = malloc(sizeof(double)*n);

                omp_set_num_threads(threads);

                int i;
                #pragma omp parallel for default(shared) private(i)
                for(i = 0; i < n; i++)
	                array[i] = (rand()+0.0) / INT_MAX;

                struct timeval start, end;
                long secs_used, micros;
                
                gettimeofday(&start, NULL);
                
                bucketSort(array, n, buckets); 

                gettimeofday(&end, NULL);
                secs_used=(end.tv_sec - start.tv_sec); //avoid overflow by subtracting first
                micros = end.tv_usec - start.tv_usec;
                double time = secs_used+0.0;
                time += ((micros+0.0) / 1000000);
                
                if(threads == 1)
                      seq_time = time;
                
                
                if(atoi(argv[3]) == 0) {
                        speedup = seq_time / time;
                } else {

                        speedup = threads * seq_time / time;
                }
                efficiency = speedup / threads;
                karp_flatt = (1. / speedup - 1. / threads) / (1. - 1. / threads); 
                printf("%d %d %lf %lf %lf %lf\n", n, threads, time, speedup, efficiency, karp_flatt);

                /*for(i = 0; i < n; i++)
	                printf("%lf ", array[i]);
                puts("\n");*/

                free(array);
        }
}
