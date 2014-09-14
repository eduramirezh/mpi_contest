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
    return [lo, hi]


class E_Ramirez():
    def get_global_minmax(self, data, comm):
        rank = comm.Get_rank()
        #boundaries[ localmin, localmax ]
        boundaries = minmax(data)
        boundaries = comm.gather(boundaries, root=0)
        result = []
        if rank == 0:
            result = minmax(itertools.chain.from_iterable(boundaries))
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
            my_sample = sorted(itertools.chain.from_iterable(my_sample))
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
                my_bucket = sorted(itertools.chain.from_iterable(this_bucket))
        my_bucket = comm.gather(my_bucket, root=0)
        if rank == 0:
            my_bucket = list(itertools.chain.from_iterable(my_bucket))
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
                my_bucket = sorted(itertools.chain.from_iterable(this_bucket))
        my_bucket = comm.gather(my_bucket, root=0)
        if rank == 0:
            my_bucket = list(itertools.chain.from_iterable(my_bucket))
        return my_bucket


    def sparse_graph_coloring(self, vector, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        colors = []
        neighbors = []
        finished_neighbors = {}
        neighbors_colors = {}
        for i in range(size):
            if vector[i]:
                neighbors.append(i)
                finished_neighbors[i] = False
                neighbors_colors[i] = 0
        deltas = [0 for i in range(size)]
        deltas[rank] = len(neighbors)
        deltas = comm.alltoall(sendobj=deltas)
        largest_delta = max(deltas)
        finished = False
        my_color = numpy.random.randint(0, largest_delta + 1)
        while not finished:
            for i in range(size):
                if rank == i:
                    for n in neighbors:
                        comm.send(my_color, dest=n)
                elif i in neighbors:
                    neighbors_colors[i] = comm.recv(source=i)
            finished = True
        colors = comm.gather(my_color, root=0)
        return colors

    def update(self, data, vector, my_neighbors, comm):
        queue = data[0]
        visited = data[1]
        cumulative = data[2]
        rank = comm.Get_rank()
        my_value = cumulative[rank]
        for n in my_neighbors:
            if n not in visited and n not in queue:
                queue.append(n)
            if cumulative[n] > my_value + vector[n]:
                cumulative[n] = my_value + vector[n]
        if len(queue) > 0:
            queue.sort(reverse=True, key=lambda q: cumulative[q])
            next_node = queue.pop()
            visited.append(next_node)
            comm.send((queue, visited, cumulative), dest = next_node, tag = 1)
        else:
            comm.send(cumulative, tag=2, dest=0 )
        return cumulative

    def shortest_paths(self, vector, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        my_neighbors = []
        for i in range(size):
            if vector[i] == 0:
                vector[i] = 9223372036854775807
            else:
                my_neighbors.append([vector[i],i])
        my_neighbors.sort(reverse = True)
        my_neighbors = [x for y, x in my_neighbors]
        vector[rank] = 0
        status = MPI.Status()
        if rank == 0:
            visited = [rank]
            queue = []
            queue.extend(my_neighbors)
            next_node = queue.pop()
            visited.append(next_node)
            to_send= queue, visited, vector
            comm.send(to_send, dest = next_node, tag = 1)
            vector = comm.recv(tag = 2,source=MPI.ANY_SOURCE, status=status )
        else:
            vector = self.update(comm.recv(tag=1, source=MPI.ANY_SOURCE, status=status), vector, my_neighbors, comm)
        return vector

