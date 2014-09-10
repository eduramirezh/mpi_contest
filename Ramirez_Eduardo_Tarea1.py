from mpi4py import MPI
import numpy
import itertools
import bisect
import random
from collections import deque

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



    def sparse_graph_sort(self, vector, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        neighbors = set()
        for i in range(size):
            if vector[i]:
                neighbors.add(i)
        finished_neighbors = set()
        deltas = comm.gather(len(neighbors), root=0)
        largest_delta = 0
        if(rank == 0):
            largest_delta = max(deltas)
        largest_delta = comm.bcast(largest_delta, root=0)
        two_delta = 2 * largest_delta
        colors = [ 0 for i in range(size)]
        final_nc = []
        temporary_nc = []
        my_final_color = -1
        while my_final_color < 0:
            temporary_nc = []
            my_temp = random.randint(0, two_delta)
            for i in neighbors:
                comm.send(my_temp, dest=i, tag=0)
            for i in (neighbors - finished_neighbors):
                temporary_nc.append(comm.recv(source = i, tag = 0))
            if my_temp not in (temporary_nc + final_nc):
                my_final_color = my_temp
                #send final
        return colors

    def update(self, data, vector, my_neighbors, comm):
        queue = data[0]
        visited = data[1]
        cumulative = data[2]
        rank = comm.Get_rank()
        size = comm.Get_size()
        my_value = cumulative[rank]
        for n in my_neighbors:
            if not visited[n] and not queue[n]:
                queue[n] = True
            if cumulative[n] > my_value + vector[n]:
                cumulative[n] = my_value + vector[n]
        next_node = -1
        for i in range(size):
            if queue[i] and (next_node == -1 or cumulative[i] < cumulative[next_node]):
                next_node = i
        if next_node > -1:
            visited[next_node] = True
            queue[next_node] = False
            comm.send((queue, visited, cumulative), dest = next_node, tag = 1)
        else:
            comm.send(cumulative, tag=2, dest=0 )
        return cumulative

    def shortest_path(self, vector, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        my_neighbors = []
        visited = numpy.array([False for i in range(size)], dtype=bool)
        for i in range(size):
            if vector[i] == 0:
                vector[i] = 2147483647
            else:
                my_neighbors.append(i)
        vector[rank] = 0
        status = MPI.Status()
        if rank == 0:
            visited[rank] = True
            queue = numpy.array([False for i in range(size)], dtype=bool)
            for n in my_neighbors:
                queue[n] = True
                next_node = n
            for i in range(size):
                if queue[i] and vector[i] < vector[next_node]:
                    next_node = i
            visited[next_node] = True
            queue[next_node] = False
            to_send= queue, visited, vector
            comm.send(to_send, dest = next_node, tag = 1)
            vector = comm.recv(tag = 2,source=MPI.ANY_SOURCE, status=status )
        else:
            vector = self.update(comm.recv(tag=1, source=MPI.ANY_SOURCE, status=status), vector, my_neighbors, comm)
        return vector

