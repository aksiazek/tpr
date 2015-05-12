#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

#define N 200000

struct bucket {
	int count;
	int* values;
};

int compareIntegers(const void* first, const void* second)
{
    int a = *((int*)first), b =  *((int*)second);
    if (a == b)
    {
        return 0;
    }
    else if (a < b)
    {
        return -1;
    }
    else
    {
        return 1;
    }
}

void bucketSort(int array[], int n, int bucket_count)
{
    struct bucket* buckets = malloc(sizeof(struct bucket)*bucket_count);
    int i, j, k;
    for (i = 0; i < bucket_count; i++)
    {
        buckets[i].count = 0;
	buckets[i].values = malloc(sizeof(int)*N);
    }
   
    for (i = 0; i < n; i++)
    {
        int step = INT_MAX / bucket_count - INT_MIN / bucket_count;
	int left_range = INT_MIN; int right_range = left_range+step;
	j = 0;
	int already = 0;
	while(!already) {
		if((array[i] >= left_range) && (array[i] <= right_range)) {
			already = 1;
			buckets[j].values[buckets[j].count++] = array[i];
		}
		j++;
		left_range+=step;
		right_range+=step;

	}
        
    }

    #pragma omp parallel for default(shared) private(i, j, k)
    for (i = 0; i < bucket_count; i++)
    {
	k = 0;
        qsort(buckets[i].values, buckets[i].count, sizeof(int), &compareIntegers);
        for (j = 0; j < buckets[i].count; j++)
        {
            array[k + j] = buckets[i].values[j];
        }
        k += buckets[i].count;
	free(buckets[i].values);
    }
    free(buckets);
}

int main()
{
	int* array = malloc(sizeof(int)*N);
	srand(time(NULL));
	for(int i = 0; i < N; i++)
		array[i] = rand();

        time_t start = time(NULL);
        
	bucketSort(array, N, 100); 
	
        time_t end = time(NULL) - start;

	

	for(int i = 0; i < N; i++)
		printf("%d ", array[i]);
	puts("\n");

	free(array);
}
