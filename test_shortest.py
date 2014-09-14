from mpi4py import MPI
from Ramirez_Eduardo_Tarea1 import E_Ramirez
import time
import numpy as np
import scipy.sparse.csgraph


def test_shortest():
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size = comm.Get_size()

    if rank==0:
        data = np.random.randint(size*2, size=(size, size))
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

    result = yourobject.shortest_paths(data, comm)
    if rank == 0:
        end = time.clock()
        real_result = scipy.sparse.csgraph.dijkstra(original, directed = False)[0]
        passed = np.array_equal(result, real_result)
        print('time: ', '%.5f' % (end - start), ' passed: ', passed)

if __name__=="__main__":
    for i in range(1000):
        test_shortest()


