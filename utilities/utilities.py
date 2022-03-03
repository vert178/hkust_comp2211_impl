import numpy as np
import random

# Generates data for results. Last column being the results, other columns being the condition of the i-th symptom
# ratio_dict: 2D array of tuples. First dimension: result_i, second dimension: evidence_i
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


# Turns string data data into discrete data:
def Relabel(data: np.ndarray):
    for i in range(data.shape[1]):
        v = np.vectorize(lambda x: np.where(np.unique(data[:, i]) == x)[0][0])
        data[:, i] = v(data[:, i])
    return data


### Testing ###
if __name__ == "__main__":
    a = np.array([
        [1, 2, 2, "a"],
        [1, 1, 1, "b"],
        [2, 2, 1, "a"],
        [1, 1, 1, "c"],
        [2, 1, 1, "b"],
    ])
    Relabel(a)
    print(a)