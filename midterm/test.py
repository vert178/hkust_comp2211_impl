from ml import Timer
from Bayes import Bayes
from knn import KNearestNeighbour
from kmeans import KmeansClustering
from neuralnet import MultiLayerPerceptron
import numpy as np
import random

# Small data test

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
    assert abs(cumulative - 1) < 1e-5
    return result

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

ratios = np.array([
    [(0.1, 0.2, 0.7), (0.7, 0.1, 0.2), (0.5, 0.5), (0.1, 0.9)],
    [(0.2, 0.3, 0.5), (0.2, 0.4, 0.4), (0.1, 0.9), (0.6, 0.4)],
    [(0.7, 0.3, 0.0), (0.3, 0.6, 0.1), (0.2, 0.8), (0.2, 0.8)],
    [(0.5, 0.3, 0.2), (0.4, 0.2, 0.4), (0.7, 0.3), (0.7, 0.3)], 
    [(0.0, 0.7, 0.3), (0.4, 0.5, 0.1), (0.8, 0.2), (0.2, 0.8)], 
], dtype = object)

train_data = GenerateData(ratios, count = 6000)
test_data = GenerateData(ratios, count = 10)
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, : -1]
y_test = test_data[:, -1]

# Print test data if it is small, for debug
if y_test.shape[0] < 15:
    print(test_data)

# bayes_timer = Timer(Bayes(1), X_train, y_train, X_test, y_test)
# knn_timer = Timer(KNearestNeighbour(5), X_train, y_train, X_test, y_test)
# kmc_timer = Timer(KmeansClustering(), X_train, y_train, X_test, y_test)
mlp = MultiLayerPerceptron((len(ratios[0]), len(ratios)), (2, 2), 100)
ann_timer = Timer(mlp, X_train, y_train, X_train, y_train)