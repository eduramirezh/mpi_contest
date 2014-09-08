from mpi4py import MPI
import numpy
import itertools
import bisect

def minmax(data):
    'Computes the minimum and maximum values in one-pass using only 1.5*len(data) comparisons'
    it = iter(data)
    try:
        lo = hi = next(it)
    except StopIteration:
        raise ValueError('minmax() arg is an empty sequence')
    for x, y in itertools.zip_longest(it, it, fillvalue=lo):
        if x > y:
            x, y = y, x
        if x < lo:
            lo = x
        if y > hi:
            hi = y
    return lo, hi


class E_Ramirez():
    def get_global_minmax(self, data, comm):
        rank = comm.Get_rank()
        #boundaries[ localmin, localmax ]
        boundaries = minmax(data)
        boundaries = comm.gather(boundaries, root=0)
        result = []
        if rank == 0:
            min_list, max_list = zip(*boundaries)
            result.append(min(min_list))
            result.append(max(max_list))
        result = comm.bcast(result, root=0)
        return result


    def define_separators(self, data, n_samples, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        #every process gets an equal size subarray from where to get a sample
        num_elements_to_select = int(n_samples/size)
        if(n_samples%size > rank):
            num_elements_to_select += 1
        my_sample = numpy.random.choice(data, num_elements_to_select, replace=False)
        my_sample = comm.gather(my_sample, root=0)
        #select m-1 separators
        separators = []
        if(rank == 0):
            my_sample = sorted([item for sublist in my_sample for item in sublist])
            sub_arrays = numpy.array_split(my_sample, size)
            separators = [x[-1] for x in sub_arrays[:-1]]
        separators = comm.bcast(separators, root=0)
        return separators

    def sample_sort(self, data, n_samples, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        #firstly, get separators
        separators = self.define_separators(data, n_samples, comm)
        buckets = [[] for i in range(size)]
        for d in data:
            buckets[bisect.bisect(separators, d)].append(d)
        my_bucket = []
        for s in range(size):
            this_bucket = buckets[s]
            this_bucket = comm.gather(this_bucket, root=s)
            if rank == s:
                my_bucket = sorted([item for sublist in this_bucket for item in sublist])
        my_bucket = comm.gather(my_bucket, root=0)
        if rank == 0:
            my_bucket = [item for sublist in my_bucket for item in sublist]
        return my_bucket



    def sparse_graph_sort(self, matrix, numberOfProcesses):
        return matrix

    def shortest_path_sort(self, matrix, s, numberOfProcesses):
        return matrix

    def bucket_sort(self, data, comm):
        a, b = self.get_global_minmax(data, comm)
        rank = comm.Get_rank()
        size = comm.Get_size()
        highest_rank = size - 1
        buckets = [[] for i in range(size)]
        fraction = (b-a)/size
        for d in data:
            current = int((d - a)/fraction)
            if (current > highest_rank):
                current = highest_rank
            buckets[current].append(d)
        my_bucket = []
        for s in range(size):
            this_bucket = buckets[s]
            this_bucket = comm.gather(this_bucket, root=s)
            if rank == s:
                my_bucket = sorted([item for sublist in this_bucket for item in sublist])
        my_bucket = comm.gather(my_bucket, root=0)
        if rank == 0:
            my_bucket = [item for sublist in my_bucket for item in sublist]
        return my_bucket



