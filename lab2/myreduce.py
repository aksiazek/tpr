#!/usr/bin/env python

from mpi4py import MPI

def receive_message(source):
	status = MPI.Status()
	data = comm.recv(source=source, status=status)
	return data

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
	if rank == 0:
		for destination in xrange(1, size):
			comm.send(number, dest=destination)
		sum = number
		for source in xrange(1, size):
			sum += receive_message(source)
	else:
		data = receive_message(0)
		comm.send(data, dest=0)

end_time = MPI.Wtime()

if rank == 0:
	print (end_time - start_time)/number_of_messages
	#print sum

MPI.Finalize()