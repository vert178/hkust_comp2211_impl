import numpy as np
import random

# Written with minimum dependencies in mind. Not really optimized in terms of memory or runtime
# Discrete values only for now


def list_find(lis, data):
    try:
        return lis.index(data)
    except ValueError as err:
        return -1


__EPSILON = 1e-5


def gen_data(t: tuple):
    rand = random.random()
    cumulative = 0
    c2 = 0
    result = 0
    for i in range(len(t)):
        assert type(t[i]) == float or type(t[i]) == int
        cumulative += t[i]
        if i > 0:
            c2 += t[i - 1]

        if rand >= c2 and rand < cumulative:
            result = i

    assert abs(cumulative - 1) < __EPSILON
    return result


class k_Naive_Bayes():

    # Evidence arr: 2D array containing "symptoms"
    # Result arr: 1D array containing results
    def __init__(self, evidence_arr: np.ndarray, result_arr: np.ndarray):

        # Get an array of all the unique results
        self.__results = np.unique(result_arr)

        # Create a list of all the possible result, row pairs
        lookup_list = []
        for i in range(evidence_arr.shape[1]):
            lookup_list += [(i, val) for val in np.unique(evidence_arr[:, i])]

        # Turns the data into a bool data array-ish thing to flag presence of some data
        data_count = evidence_arr.shape[0]
        self.__data = np.zeros((data_count, len(lookup_list) + 1), dtype=int)

        # Loops through all the variable combinations
        for i in range(len(lookup_list)):
            pair = lookup_list[i]
            self.__data[:, i] = evidence_arr[:, pair[0]] == pair[1]
        
        self.__data[:, -1] = result_arr
        self.__lookup_list = lookup_list

    def predict(self, evidence, alpha=1):
        assert alpha > 0
        result = []
        prediction = (0, -1e99)

        evid_arr = np.zeros((len(self.__lookup_list), ), dtype = bool)
        for i in range(len(self.__lookup_list)):
            pair = self.__lookup_list[i]
            evid_arr[i] = evidence[pair[0]] == pair[1]

        # P(Ri | evidences) = P(Ri and evidences) / junk
        # Take log for good measures

        for k in self.__results:
            result_k = self.__data[:, -1] == k
            P = np.log(np.count_nonzero(result_k)/len(self.__data) + alpha)
            filtered_results = self.__data[result_k][:, :-1]
            count = np.sum(filtered_results, axis = 0) + alpha
            filtered_count = count[evid_arr]
            P += np.sum(np.log(filtered_count / filtered_results.shape[0]))

            result += (k, P)
            if P > prediction[1]:
                prediction = (k, P)

        return result, prediction

    # Generates data for results
    # ratio_dict: 2D array of tuples. First dimension: result_i, second dimension: evidence_i

    @staticmethod
    def GenerateData(ratio_dict, count=250):
        count = int(count)
        num_results = len(ratio_dict)
        num_evidence = len(ratio_dict[0])
        data = np.zeros((count, num_evidence + 1), dtype=int)

        for i in range(count):
            result = random.randint(0, num_results - 1)
            data[i][-1] = result
            for j in range(len(ratio_dict[result])):
                t = ratio_dict[result][j]
                assert type(t) == tuple
                data[i][j] = gen_data(t)

        return np.array(data)
