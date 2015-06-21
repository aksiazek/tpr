#!/usr/bin/env python

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

number = 0
sum = 0
number_of_messages = 2000

if rank == 0:
	number = 1
	
comm.Barrier()
	
start_time = MPI.Wtime()

for _ in xrange(0, number_of_messages):
	number = comm.bcast(number, root=0)
	sum = comm.reduce(number, sum, op=MPI.SUM, root=0)

end_time = MPI.Wtime()

if rank == 0:
	print (end_time - start_time)/number_of_messages
	#print sum
	
MPI.Finalize()