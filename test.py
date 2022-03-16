from bayes import k_Naive_Bayes as Bayes
from knn import K_nearest_neighbour as knn
from simple_perceptron import Perceptron as perceptron
import numpy as np
import time

def Bayes():
    # This means there are 4 different result and 4 different "symptoms". If it turns out that the result is 0, then there is a 0.1 chance that symptom 0 will have 0.1 chance being 0, 0.2 chance being 1, 0.7 chance being 2, and so on
    ratios = np.array([
        [(0.1, 0.2, 0.7), (0.7, 0.1, 0.2), (0.5, 0.5), (0.1, 0.9)],
        [(0.2, 0.3, 0.5), (0.2, 0.4, 0.4), (0.1, 0.9), (0.6, 0.4)],
        [(0.7, 0.3, 0.0), (0.3, 0.6, 0.1), (0.2, 0.8), (0.2, 0.8)],
        [(0.5, 0.3, 0.2), (0.4, 0.2, 0.4), (0.7, 0.3), (0.7, 0.3)], 
        [(0.0, 0.7, 0.3), (0.4, 0.5, 0.1), (0.6, 0.4), (0.4, 0.6)], 
    ], dtype = object)

    data = Bayes.GenerateData(ratios, count = 8000)
    b = Bayes(data[:,:-1], data[:,-1])
    result, prediction = b.predict([2, 2, 1, 1])
    print(prediction)

def KNN():
    data = knn.GenerateData((
        (1, 2, 3, 4, 5),
        (10, 11, 12),
        ("hehe", "haha", "hihi", "hoho", "huhu")
    ), 1000)

    k = knn(data)
    results = k.Get_knn(3, [1, 10, "haha"])
    print(results)

def Perceptron():
    def secret_function(x, y, z):
        return 42 * x + y - 69
    
    data_count = 1000
    test_count = 100
    trainX = np.random.randint(0, 15, (data_count, 3))
    trainY = secret_function(trainX[:, 0], trainX[:, 1], trainX[:, 2])
    testX = np.random.randint(0, 15, (10, 3))
    testY = secret_function(testX[:, 0], testX[:, 1], testX[:, 2])

    print("Perceptron running")
    print("Number of test data: " + str(test_count))
    
    p = perceptron(k = 3)
    p.learn(trainX, trainY, maxIter = 100000)
    predictY = p.predict(testX)

    accuracy = np.sum(np.abs(predictY - testY)) / test_count

    # Average error
    print(accuracy)

#########################################################################################################


test_func = Perceptron

def Run_Test(*args, **kwargs):
    t1 = time.time()
    test_func(*args, **kwargs)
    t2 = time.time()
    print("Benchmark: " + str(round(t2 - t1, 5)) + " seconds")

if __name__ == "__main__":
    Run_Test()