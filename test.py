import numpy as np
from Ramirez_Eduardo_Tarea1 import E_Ramirez
from mpi4py import MPI
import time

def test_bucket(numberOfProcesses, rank, numberOfElements):
    testInstance = E_Ramirez()
    if (rank == 0):
        start = time.clock()
    testInstance.bucket_sort2(np.random.randint(1,1000000,numberOfElements), numberOfProcesses, 1, 1000000)
    if (rank == 0):
        end = time.clock()
        return '%.5f' % (end - start)

def test_bucket_all(numberOfProcesses, rank):
    if (rank == 0):
        print('Bucket ', numberOfProcesses, ' processes '),
    results = []
    results.append(test_bucket(numberOfProcesses, rank, 1000))
    results.append(test_bucket(numberOfProcesses, rank, 10000))
    results.append(test_bucket(numberOfProcesses, rank, 100000))
    results.append(test_bucket(numberOfProcesses, rank, 1000000))
    if (rank == 0):
        print(' '.join(results))
def test_sample():
    pass
def test_sparse():
    pass
def test_shortest():
    pass

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    numberOfProcesses = comm.Get_size()
    rank = comm.Get_rank()
    test_bucket_all(numberOfProcesses, rank)
    test_sample()
    test_sparse()
    test_shortest()
