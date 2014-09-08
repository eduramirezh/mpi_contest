import numpy as np
from Ramirez_Eduardo_Tarea1 import E_Ramirez
from mpi4py import MPI
import time

def test_bucket(number_of_processes, rank, number_of_elements):
    testInstance = E_Ramirez()
    if (rank == 0):
        start = time.clock()
    testInstance.bucket_sort(np.random.randint(1,1000000,number_of_elements), number_of_processes, 1, 1000000)
    if (rank == 0):
        end = time.clock()
        return '%.5f' % (end - start)

def test_bucket_all(number_of_processes, rank):
    if (rank == 0):
        print('Bucket ', number_of_processes, ' processes '),
    results = []
    results.append(test_bucket(number_of_processes, rank, 1000))
    results.append(test_bucket(number_of_processes, rank, 10000))
    results.append(test_bucket(number_of_processes, rank, 100000))
    results.append(test_bucket(number_of_processes, rank, 1000000))
    if (rank == 0):
        print(' '.join(results))

def test_sample(number_of_processes, rank , number_of_elements, s, m):
    testInstance = E_Ramirez()
    if (rank == 0):
        start = time.clock()
    testInstance.sample_sort(np.random.randint(1,1000000,number_of_elements), number_of_processes, s, m)
    if (rank == 0):
        end = time.clock()
        return '%.5f' % (end - start)

def test_sample_all(number_of_processes, rank):
    if(rank == 0):
        print('Bucket ', number_of_processes, ' processes '),
    results = []
    results.append(test_sample(number_of_processes, rank, 1000, 100, 5))
    results.append(test_bucket(number_of_processes, rank, 10000, 1000, 5))
    results.append(test_bucket(number_of_processes, rank, 100000, 10000, 5))
    results.append(test_bucket(number_of_processes, rank, 1000000, 100000))
    if (rank == 0):
        print(' '.join(results))

def test_sparse():
    pass
def test_shortest():
    pass

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    number_of_processes = comm.Get_size()
    rank = comm.Get_rank()
    #test_bucket_all(number_of_processes, rank)
    test_sample(number_of_processes, rank, 10000, 100, 5)
    test_sparse()
    test_shortest()
