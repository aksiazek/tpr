#!/usr/bin/python

from math import tan, exp, log

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

number_of_messages = 2000


def computations():
    x = 123456789
    for i in range(100):
        x = x * x % 23456789


comm.Barrier()

start_time = MPI.Wtime()

for _ in xrange(0, number_of_messages):
    comm.Barrier()
    computations()

end_time = MPI.Wtime()

elapsed_barrier = end_time - start_time

start_time = MPI.Wtime()

for _ in xrange(0, number_of_messages):
    computations()

end_time = MPI.Wtime()

elapsed_no_barrier = end_time - start_time

if rank == 0:
    print "Barrier: {0} [s], no barrier: {1} [s]".format(elapsed_barrier, elapsed_no_barrier)
