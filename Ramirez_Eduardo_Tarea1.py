from mpi4py import MPI
import numpy
import itertools

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


    def define_separators(self, elements, number_of_processes, s, m, comm):
        rank = comm.Get_rank()
        number_of_elements = len(elements)
        last_rank = number_of_processes - 1
        #every process gets an equal size subarray from where to get a sample
        fraction = number_of_elements/number_of_processes
        bottom = int(fraction * rank)
        if (rank == last_rank):
            top = number_of_elements
        else:
            top = int(fraction*(rank + 1))
        num_elements_to_select = int(s/number_of_processes)
        if(s%number_of_processes > rank):
            num_elements_to_select += 1
        my_sample = numpy.random.choice(elements[bottom:top], num_elements_to_select, replace=False)
        my_sample.sort()
        comm.gather(my_sample, root=last_rank)
        #select m-1 separators
        my_separators = []
        if(rank == last_rank):
            my_sample.sort()
            sub_arrays = numpy.array_split(my_sample, m)
            separators = [x[-1] for x in sub_arrays]
            my_separators = [separators[-2], -1]
            #send message to the rest of the processes
            for i in range(last_rank):
                if (i != 0):
                    to_send = [separators[i-1], separators[i]]
                else:
                    to_send = [-1, separators[i]]
                comm.send(to_send, dest=i)
        else:
            my_separators = comm.recv(source=last_rank)
        return my_separators

    def sample_sort(self, elements, number_of_processes, s, m):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        #firstly, get separators
        my_separators = self.define_separators(elements, number_of_processes, s, m, comm)
        my_bucket = []
        for e in elements:
            if((e >= my_separators[0] or e == -1) and (e < my_separators[1] or my_separators[1] == -1)):
                    my_bucket.append(e)
        my_bucket.sort()
        return comm.gather(my_bucket, root=0)



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



