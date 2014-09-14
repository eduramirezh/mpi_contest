from mpi4py import MPI
from Ramirez_Eduardo_Tarea1 import E_Ramirez
import time
import numpy as np
import scipy.sparse.csgraph


def test_sparse():
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size = comm.Get_size()
    if rank==0:
        data = np.ndarray(shape=(size, size), dtype=bool)
        for i in range(size):
            for j in range(i + 1):
                if i == j:
                    data[i][j] = 0
                data[j][i] = data [i][j]
        original = data
    else:
        data=None
    data = comm.scatter(data,root=0)
    yourobject=E_Ramirez()
    if rank == 0:
        start = time.clock()

    result = yourobject.sparse_graph_coloring(data, comm)
    if rank == 0:
        end = time.clock()
        print(result)
        print('time: ', )
        print( '%.5f' % (end - start))

if __name__=="__main__":
    test_sparse()


