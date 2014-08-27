import numpy as np
from mpi4py import MPI

class E_Ramirez():
    def bucket_sort(self, elements, numberOfProcesses, a, b):
        fraction = (b-a)/numberOfProcesses
        buckets = [[] for x in xrange(numberOfProcesses)]
        for e in elements:
            if (e-a)/fraction >= numberOfProcesses:
                buckets[numberOfProcesses-1].append(e)
            else:
                buckets[(e-a)/fraction].append(e)
        return buckets

    def sample_sort(self, elements, numberOfProcesses):
        return elements

    def sparse_graph_sort(self, matrix, numberOfProcesses):
        return matrix

    def shortest_path_sort(self, matrix, s, numberOfProcesses):
        return matrix
