import numpy as np
import random


# Data generation algorithms
__EPSILON = 1e-5
PI = 3.1415926


def isNum(val):
    try:
        int(val)
        return True
    except:
        return False


# Default metric
def L2norm(x, y):
    square_sum = 0
    if isNum(x) and isNum(y):
        square_sum += (x - y) ** 2
    elif type(x) == str and type(y) == str:
        # Trivial metric: 1 if not equal, else 0
        square_sum += int(x != y)
    return square_sum ** 0.5


class K_nearest_neighbour():
    # Give it some 1D-array of data, either number type or some other discrete type
    def __init__(self, data: np.ndarray):
        self.__data = np.array(data, dtype = object)

    # k is the k in k nearest neighbour
    # When left unspecified, metric will be Euclidean distance

    def Get_knn(self, k: int, predict_data, metric=L2norm):
        assert k > 0
        pd = np.array(predict_data, dtype = object)
        return self.Recursive_get_knn(k, pd, self.__data, metric, 0)

    def Recursive_get_knn(self, k: int, predict_data, data, metric=L2norm, iteration=0):
        metric = np.vectorize(metric)

        # God bless broadcasting
        dist_arr = np.array(metric(data, predict_data))
        distances = np.sum(dist_arr, axis=1)

        # Now sort array: return the smallest element
        closest = data[np.argmin(distances)]

        # Remove the closest element from the array, for recursion magic to happen
        data = data[np.any(data != closest, axis = 1)]
        # Increments iteration and feed in the new data
        if iteration < k - 1:
            res = list(self.Recursive_get_knn(k, predict_data, data, metric, iteration + 1))
            res.append(list(closest))
            return res
        else :
            return [list(closest)]

    # Generate data with count
    # available_data: tuple of tuples containing the data

    @staticmethod
    def GenerateData(available_data: tuple, count=250):
        data = np.zeros((count, len(available_data)), dtype=object)
        for j in range(len(available_data)):
            data[:, j] = np.random.choice(available_data[j], count)
        return data