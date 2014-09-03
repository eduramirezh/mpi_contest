from mpi4py import MPI

class E_Ramirez():
    def sample_sort(self, elements, numberOfProcesses, s, m):
        #firstly, get separators
        numberOfElements = len(elements)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        #every process gets an equal size subarray from where to get a sample
        my_sample = []
        fraction = numberOfElements/numberOfProcesses
        bottom = int(fraction*rank)
        if (rank == numberOfProcesses - 1):
            top = numberOfElements
        else:
            top = int(fraction*(rank + 1))
        for e in elements[bottom:top]:
            pass
        return elements

    def sparse_graph_sort(self, matrix, numberOfProcesses):
        return matrix

    def shortest_path_sort(self, matrix, s, numberOfProcesses):
        return matrix

    def bucket_sort(self, elements, numberOfBuckets, a, b):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        my_bucket = []
        fraction = (b-a)/numberOfBuckets
        bottom = a + int(fraction*rank)
        if (rank == numberOfBuckets - 1):
            top = b + 1
        else:
            top = a + int(fraction*(rank + 1))
        for e in elements:
            if e >= bottom and e < top:
                my_bucket.append(e)
        my_bucket.sort()
        return comm.gather(my_bucket, root=0)



