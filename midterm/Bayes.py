from ml import ML, Timer
import numpy as np

# This is just a quick implementation to check my concepts. While it is not built with the most generality in mind,
# It is built with a bit of speed optimization in mind, utilizing numpy functions whenever possible
class Bayes(ML):
    def __init__(self, alpha = 1e-10):
        self.alpha = alpha
    
    def train(self, X, y):
        # The formula is P(B | e) = P(B) P(e1 | B) P(e2 | B) ... / junk
        # We want to use log to prevent underflowing
        self.X = X
        self.y = y

    def predict(self, X):
        beliefs = np.unique(self.y)
        nb = beliefs.shape[0]
        results = np.zeros((X.shape[0], nb))
        for i in range(nb):
            b = beliefs[i]
            # log(P(B)) = log(#equals) - log(#totals)
            log_pbi = np.log(np.count_nonzero(self.y == b) + self.alpha) - np.log(self.y.shape[0])
            # log(P(e|B) = log(#le) - log(#filtered)
            filtered = self.X[self.y == b].reshape(1, -1, X.shape[1])
            X2 = X.reshape(-1, 1, X.shape[1])
            nf = filtered.shape[1]
            # Embrace third dimensional broadcasting magic
            # peb is an array with shape that of X, while log_pbi is a number
            log_peb = np.log(np.count_nonzero(filtered == X2, axis = 1) + self.alpha) - np.log(nf + self.alpha)
            log_peb = np.sum(log_peb, axis = 1)
            results[:, i] = log_peb + log_pbi
        
        # count along inner axis to get the max, and then do integer array indexing
        best_result = np.argmax(results, axis = 1)
        best_guesses = beliefs[best_result]
        return best_guesses