from ml import ML, Timer
import numpy as np

class KNearestNeighbour(ML):
    def __init__(self, k: int):
        self.k = k
    
    def train(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # nf = number of features
        nf = X.shape[1]
        outputs = np.unique(self.y)
        nb = outputs.shape[0]
        
        Xtrain = self.X.reshape(1, -1, X.shape[1])
        Xtest = X.reshape(-1, 1, X.shape[1])
        # I realized you probably can get away with not taking log since
        # The maximum distance you will get is like 8 digits only, far from overflow
        dists_sum = np.sum((Xtrain - Xtest) ** 2, axis = 2)
        best_result = np.argsort(dists_sum, axis = 1)[:, :self.k]
        # Use integer array indexing to get all the ys
        best_ys = self.y[best_result]
        # Now do majority voting
        # Again, this handles discrete values only
        counts = np.zeros((X.shape[0], nb))
        
        # Ouch one for loop, oh well
        for i in range(nb):
            output = outputs[i]
            count = np.count_nonzero(best_ys == output, axis = 1)
            counts[:, i] = count
        
        sorted_votes = np.argsort(counts, axis = 1)

        # Return a random one if a tie occur lmao. This doesn't matter, unlike PA1!!
        # argsort sorts thing from smallest to largest, so most votes is at the end
        result = outputs[sorted_votes[:, -1]]
        
        return result