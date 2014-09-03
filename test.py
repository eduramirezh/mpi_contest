import numpy as np
from Ramirez_Eduardo_Tarea1 import E_Ramirez
from mpi4py import MPI
import time

def test_bucket(numberOfProcesses, rank):
    testInstance = E_Ramirez()
    if (rank == 0):
        print('1000 elements, ', numberOfProcesses, ' processes')
        start = time.clock()
    testInstance.bucket_sort(np.random.randint(5000,30000,1000), numberOfProcesses, 5000, 30000)
    if (rank == 0):
        end = time.clock()
        print(end - start)
    if (rank == 0):
        print('1000000 elements, ', numberOfProcesses, ' processes')
        start = time.clock()
    testInstance.bucket_sort(np.random.randint(5000,300000,1000000), numberOfProcesses, 5000, 30000)
    if (rank == 0):
        end = time.clock()
        print(end - start)

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
    test_bucket(numberOfProcesses, rank)
    test_sample()
    test_sparse()
    test_shortest()
