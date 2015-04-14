#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>
#include <gmp.h>

double monte_carlo_pi(gmp_randstate_t RNG_state, mpz_t all_points) {
        mpz_t in_circle_hits, i, one;
        mpf_t x, y, R, R_squared, result;
        
        mpz_init(in_circle_hits);
        mpz_init(one);
        mpf_init(x);
        mpf_init(y);
        mpf_init(R);
        mpf_init(R_squared);
        mpf_init(result);
        mpz_set_ui(one, 1);
        mpf_set_d (R_squared, 0.25);
        mpf_set_d (R, 0.5);
 
	for (mpz_init(i); mpz_cmp(i, all_points); mpz_add(i, i, one)) {
                // as in IEEE 754
                mpf_urandomb (x, RNG_state, 52); 
                mpf_urandomb (y, RNG_state, 52);
                mpf_sub(x, x, R);
                mpf_sub(y, y, R);
                mpf_mul(x, x, x);
                mpf_mul(y, y, y);
		mpf_add(result, x, y);
		if (mpf_cmp(result, R_squared) < 1)
			mpz_add(in_circle_hits, in_circle_hits, one);
	}
	mpf_set_z(x, in_circle_hits);
	mpf_set_z(y, all_points);
	mpf_div(result, x, y);
	double pi_4 = mpf_get_d(result);
	
        mpz_clear(i);
        mpz_clear(in_circle_hits);
        mpz_clear(one);
        mpf_clear(x);
        mpf_clear(y);
        mpf_clear(R);
        mpf_clear(R_squared);
        mpf_clear(result);
	return 4 * pi_4;
}

int main(int argc, char** argv) {
        int world_rank, world_size;
        gmp_randstate_t RNG_state;
        
        if (argc != 3) {
		fprintf(stderr, "Usage: %s <points-per-processor>\n", argv[0]);
		exit(1);
	}
        
        mpz_t all_points;
        mpz_init(all_points);
        mpz_t all_points_seq;
        mpz_init(all_points_seq);
        mpz_set_str(all_points, argv[1], 10);
        mpz_set_str(all_points_seq, argv[1], 10);
        
        gmp_randinit_mt(RNG_state);
        
	MPI_Init(&argc, &argv);	
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	if(atoi(argv[2]) == 1) {
	        mpz_fdiv_q_ui(all_points, all_points, world_size);
	} else {
	        mpz_mul_ui(all_points_seq, all_points_seq, world_size);
	}
	        
	unsigned int seed = time(NULL) + world_rank;
        gmp_randseed_ui(RNG_state, seed);
        
        MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	double local_pi = monte_carlo_pi(RNG_state, all_points);
	double global_pi_sum;

	MPI_Reduce(&local_pi, &global_pi_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();
	double t_p = end - start;
        
        
        if (world_rank == 0) {
                start = MPI_Wtime();

	        monte_carlo_pi(RNG_state, all_points_seq);
	                
	        end = MPI_Wtime();
	}

	double t_s = end - start;

	double speedup = t_s / t_p;
	double efficiency = speedup / world_size;
	double karp_flatt = (1. / speedup - 1. / world_size) / (1. - 1. / world_size);

        
        if (world_rank == 0) {
		fprintf(stderr, "world_size = %i, all_points = %lu, ", world_size, mpz_get_ui(all_points_seq));
		fprintf(stderr, "pi = %lf, ", global_pi_sum / world_size);
		fprintf(stderr, "t_s = %lf sec, t_p = %lf sec, ", t_s, t_p);
		fprintf(stderr, "speedup = %lf, efficiency = %lf, karp_flatt = %lf\n", 
		        speedup, efficiency, karp_flatt);
		
		fprintf(stdout, "%i %lf %lf %lf %lf\n", world_size, 
		        t_p/world_size, speedup, efficiency, karp_flatt);
	}
        
        mpz_clear(all_points);
        mpz_clear(all_points_seq);
        gmp_randclear(RNG_state);
	MPI_Finalize();
	return 0;
}

