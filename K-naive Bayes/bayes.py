import numpy as np

# Written with minimum dependencies in mind. Not really optimized in terms of memory or runtime
# Discrete values only for now


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

    # Evidence is nparray of "conditions" of the "symptoms"
    def predict(self, evidence, alpha=1):
        assert alpha > 0
        result = []
        prediction = (0, -1e99)

        evid_arr = np.zeros((len(self.__lookup_list), ), dtype=bool)
        for i in range(len(self.__lookup_list)):
            pair = self.__lookup_list[i]
            evid_arr[i] = evidence[pair[0]] == pair[1]

        # P(Ri | evidences) = P(Ri and evidences) / junk
        # Take log for good measures

        for k in self.__results:
            result_k = self.__data[:, -1] == k
            P = np.log(np.count_nonzero(result_k)/len(self.__data) + alpha)
            filtered_results = self.__data[result_k][:, :-1]
            count = np.sum(filtered_results, axis=0) + alpha
            filtered_count = count[evid_arr]
            P += np.sum(np.log(filtered_count / filtered_results.shape[0]))

            result += P
            if P > prediction[1]:
                prediction = (k, P)

        # Result: list containing results
        # prediction: tuple with [0] being the indication that i-th result is best, and [1] being the log of the probability
        return result, prediction