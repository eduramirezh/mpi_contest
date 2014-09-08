from mpi4py import MPI
from Ramirez_Eduardo_Tarea1 import E_Ramirez
import numpy as np
import time

def test_sample(numberOfElements, n_samples):
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()

    if rank==0:
        data = [np.random.randint(1000,100000,int(numberOfElements/size)) for i in range(size)]
        copy = data
    else:
        data=None
    data = comm.scatter(data,root=0)
    yourobject=E_Ramirez()
    if rank == 0:
        print(numberOfElements, '-->')
        start = time.clock()

    result = yourobject.sample_sort(data, n_samples, comm)
    if rank == 0:
        end = time.clock()
        if len(result) == numberOfElements and all(b >= a for a, b in zip(result, result[1:])):
            print('test passed!')
        else:
            print('test NOT passed')
        print( '%.5f' % (end - start))

if __name__=="__main__":
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()
    test_sample(10000, 100)
    test_sample(100000, 1000)
    test_sample(1000000, 1000)


