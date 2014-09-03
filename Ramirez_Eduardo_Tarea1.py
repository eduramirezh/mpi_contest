from mpi4py import MPI

class E_Ramirez():
    def bucket_sort(self, elements, numberOfBuckets, a, b):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if (rank == 0):
            buckets = [[] for i in range(numberOfBuckets)]
            fraction = int((b-a)/numberOfBuckets)
            for e in elements:
                index = int((e-a)/fraction)
                if ( index < numberOfBuckets):
                    buckets[index].append(e)
                else:
                    buckets[numberOfBuckets-1].append(e)
        else:
            buckets = None
        localArray = sorted(comm.scatter(buckets, root=0))
        result = comm.gather(localArray, root=0)
        return result

    def sample_sort(self, elements, numberOfProcesses, s, m):
        return elements

    def sparse_graph_sort(self, matrix, numberOfProcesses):
        return matrix

    def shortest_path_sort(self, matrix, s, numberOfProcesses):
        return matrix
