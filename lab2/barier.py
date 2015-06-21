#!/usr/bin/python

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

number_of_messages = 2000

comm.Barrier()

start_time = MPI.Wtime()

for _ in xrange(0, number_of_messages):	 
	comm.Barrier()
	
end_time = MPI.Wtime()

if rank == 0:
	print (end_time - start_time)/number_of_messages