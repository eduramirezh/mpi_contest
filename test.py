import numpy as np
from Ramirez_Eduardo_Tarea1 import E_Ramirez
def test_bucket():
    testInstance = E_Ramirez()
    for l in testInstance.bucket_sort(np.random.randint(5000,30000,1000), 8, 5000, 30000):
        print len(l)

def test_sample():
    pass
def test_sparse():
    pass
def test_shortest():
    pass

if __name__ == "__main__":
    test_bucket()
    test_sample()
    test_sparse()
    test_shortest()
