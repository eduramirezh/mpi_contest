from mpi4py import MPI
from Ramirez_Eduardo_Tarea1 import E_Ramirez
import numpy as np
import time
import itertools

def test_sample(numberOfElements, n_samples):
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()

    if rank==0:
        data = [np.random.randint(1000,100000,int(numberOfElements/size)) for i in range(size)]
        copy = list(itertools.chain.from_iterable(data))
    else:
        data=None
    data = comm.scatter(data,root=0)
    yourobject=E_Ramirez()
    if rank == 0:
        start = time.clock()

    result = yourobject.sample_sort(data, n_samples, comm)
    if rank == 0:
        end = time.clock()
        passed = np.array_equal(result, sorted(copy))
        print( '%.5f' % (end - start), ' passed: ', passed)

if __name__=="__main__":
    test_sample(10000, 100)
    test_sample(100000, 1000)
    test_sample(1000000, 1000)


