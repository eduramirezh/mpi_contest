from mpi4py import MPI
import numpy

class E_Ramirez():
    def sample_sort(self, elements, number_of_processes, s, m):
        #firstly, get separators
        number_of_elements = len(elements)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        last_rank = number_of_processes - 1
        #every process gets an equal size subarray from where to get a sample
        fraction = number_of_elements/number_of_processes
        bottom = int(fraction * rank)
        if (rank == number_of_processes - 1):
            top = number_of_elements
        else:
            top = int(fraction*(rank + 1))
        num_elements_to_select = int(s/number_of_processes)
        if(s%number_of_processes > rank):
            num_elements_to_select += 1
        my_sample = numpy.random.choice(elements[bottom:top], num_elements_to_select, replace=True)
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
                to_send = [separators[i], separators[i+1]]
                comm.send(to_send, dest=i)
            print (rank, ' : send all. My separators are ', my_separators)
        else:
            my_separators = comm.recv(source=last_rank)
            print (rank, ' : Received. My separators are ', my_separators)
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



