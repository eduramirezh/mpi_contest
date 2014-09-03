from mpi4py import MPI

class E_Ramirez():
    def sample_sort(self, elements, numberOfProcesses, s, m):
        #firstly, get separators
        return elements

    def sparse_graph_sort(self, matrix, numberOfProcesses):
        return matrix

    def shortest_path_sort(self, matrix, s, numberOfProcesses):
        return matrix

    def bucket_sort(self, elements, numberOfBuckets, a, b):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        mybucket = []
        fraction = (b-a)/numberOfBuckets
        bottom = a + int(fraction*rank)
        if (rank == numberOfBuckets - 1):
            top = b + 1
        else:
            top = a + int(fraction*(rank + 1))
        for e in elements:
            if e >= bottom and e < top:
                mybucket.append(e)
        mybucket.sort()
        return comm.gather(mybucket, root=0)



