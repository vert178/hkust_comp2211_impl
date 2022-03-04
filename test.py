from bayes import k_Naive_Bayes as Bayes
from knn import K_nearest_neighbour as knn
import numpy as np

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


if __name__ == "__main__":
    KNN()