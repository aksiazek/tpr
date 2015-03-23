#!/usr/bin/env python

from mpi4py import MPI

def receive_message(source):
    status = MPI.Status()
    data = comm.recv(source=source, status=status)
    return data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

number = 1
sum = 0
number_of_messages = 2000


comm.Barrier()

start_time = MPI.Wtime()

for _ in xrange(0, number_of_messages):
    sum = comm.reduce(number, sum, op=MPI.SUM, root=0)

end_time = MPI.Wtime()

multi_time = end_time - start_time

start_time = MPI.Wtime()

for _ in xrange(0, number_of_messages):
    comm.Barrier()
    if rank == 0:
        sum = number
        for source in xrange(1, size):
            sum += receive_message(source)
    else:
        comm.send(number, dest=0)

end_time = MPI.Wtime()

my_time = end_time - start_time

if rank == 0:
    print "Multi: {0} [s], own implementation: {1} [s]".format(multi_time/number_of_messages, my_time/number_of_messages)

MPI.Finalize()