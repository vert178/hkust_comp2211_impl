from bayes import k_Naive_Bayes as Bayes
import numpy as np

ratios = np.array([
    [(0.1, 0.2, 0.7), (0.7, 0.1, 0.2), (0.5, 0.5), (0.1, 0.9)],
    [(0.2, 0.3, 0.5), (0.2, 0.4, 0.4), (0.1, 0.9), (0.6, 0.4)],
    [(0.7, 0.3, 0.0), (0.3, 0.6, 0.1), (0.2, 0.8), (0.2, 0.8)],
    [(0.5, 0.3, 0.2), (0.4, 0.2, 0.4), (0.7, 0.3), (0.7, 0.3)], 
], dtype = object)

data = Bayes.GenerateData(ratios, count = 8000)
b = Bayes(data[:,:-1], data[:,-1])
print(b.predict([2, 2, 1, 1]))