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


# I don't really see this relabelling function useful now but i really like it so I am saving it
def Relabel(data):
    # Do some relabeling for non-number type object
    for i in range(data.shape[1] - 1):
        if not isNum(data[i][0]):
            v = np.vectorize(lambda x: np.where(
                np.unique(data[:, i]) == x)[0][0])
            data[:, i] = v(data[:, i])
    return data


def GenerateNormalDistribution(mean, sd, count=1):
    # An algorithm to generate normal distribution from uniform distribution
    # Using the neat little identity cdf_normal(x) = 1/2 + 1/2erf(x/sqrt(2),
    # We can turn a random number from -1 to 1, to a number with mean 0 sd 1

    # The messy lambda is just sqrt(2) * inverse of error function, up to x ** 7
    # which should be enough since if x < 1, x ** 9 <= 1e-9
    result = 2 * np.random.rand(count) - 1
    erf_inverse = np.vectorize(lambda x: (
        PI / 2) ** 0.5 * (x + PI/12 * x**3 + 7*PI**2/480 * x**5 + 127*PI**3/40320 * x**7))
    return mean + sd * erf_inverse(result)


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