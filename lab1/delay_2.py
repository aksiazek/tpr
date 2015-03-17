#!/usr/bin/env python

from mpi4py import MPI
import sys
from array import array
import numpy
import time

def send_message_std(destination):
	comm.Send(buff, dest=destination)

def send_message_sync(destination):
	comm.Ssend(buff, dest=destination)

def send_message_buffered(destination):
	comm.Bsend(buff, dest=destination)

def receive_message(source):
	status = MPI.Status()
	comm.Recv(buff, source=source, status=status)

def send_message(destination):
	if type == 'sync':
		send_message_sync(destination)
	elif type =='buff':
		send_message_buffered(destination)
	else:
		send_message_std(destination)

def ping_pong(number_of_messages):
	if rank == 0:
		for _ in xrange(0, number_of_messages):
			send_message(1)
			receive_message(1)
	elif rank == 1:
		for _ in xrange(0, number_of_messages):
			receive_message(0)
			send_message(0)	

if len(sys.argv) < 2 and rank == 1:
	print 'Usage ./delay <std|sync|buff>'
	sys.exit(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

type = sys.argv[1]

if type == 'buff':
	mpi_buffer = numpy.arange(10000, dtype='int8')
	MPI.Attach_buffer(mpi_buffer)

buff = numpy.arange(1, dtype='int8')
number_of_messages = 2000

comm.Barrier()

start_time = MPI.Wtime()
ping_pong(number_of_messages)
end_time = MPI.Wtime()

if rank == 1:
	delay = (end_time - start_time)* 1000/(number_of_messages*2)
	print str(delay) + " [ms]"

if type == 'buff':
	MPI.Detach_buffer()
		
MPI.Finalize()
