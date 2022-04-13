from ml import ML, Timer
import numpy as np

# This is really a modified version of K means clustering
# We start with one centroid for every unique label, and then do
# the k means convergence thing. Then we label the centroids according
# to the majority vote
# Finally when a data comes in we simply calculate its distance
# with all the centroids and return the best one
class KmeansClustering(ML):
    def train(self, X, y):
        # Get all the unique outputs
        outputs = np.unique(y)
        nb = outputs.shape[0]
        # print("nb = {}".format(nb))
        
        centroids = np.zeros((nb, X.shape[1]))
        labels = np.arange(nb)
        # Pick one centroid from each output
        for i in range(nb):
            filter = X[y == outputs[i]]
            centroids[i] = filter[0]
        
        
        # Do the convergence thing while convergence criteria is not met
        converge = False
        while not converge:
            # Calculate distance between each centroid and each data point
            cen = centroids.reshape((1, nb, X.shape[1]))
            X2 = X.reshape(-1, 1, X.shape[1])
            dists = np.sum((X2 - cen) ** 2, axis = 2)
            # dists have shape 60000, 10
            # Get the coordinates of the closest centroid
            closest_centroid = labels[np.argsort(dists, axis = 1)[: ,0]]
            
            total_nudge = 0
            # Recalculate distance of centroid
            for i in range(nb):
                # Get all the columns with the closest centroid being the ith centroid
                filter = X[closest_centroid == labels[i]]
                new_centroid_pos = np.average(filter, axis = 0)
                total_nudge += np.sum(np.abs(new_centroid_pos - centroids[i]))
                centroids[i] = new_centroid_pos
            
            converge = total_nudge < 1e-2
            
        # For each centroid do a majority vote
        # Now we update the labels
        cen = centroids.reshape((1, nb, X.shape[1]))
        X2 = X.reshape(-1, 1, X.shape[1])
        dists = np.sum((X2 - cen) ** 2, axis = 2)
        closest_centroid = labels[np.argsort(dists, axis = 1)[:, 0]]
        for i in range(nb):
            # Get all the columns with the closest centroid being the ith centroid
            # filter has shape (smth, 10)
            filter = y[closest_centroid == labels[i]].reshape(-1, 1)
            res = outputs.reshape(1, -1)
            counts = np.count_nonzero(filter == res, axis = 0)
            labels[i] = outputs[np.argmax(counts)]

        # Label all centroids and save the coordinates/labels of the centroids
        # Celebrate        
        self.labels = labels
        self.centroids = centroids
        
    def predict(self, X):
        # The cen statement implicitly asserts X must have the same number of features as centroids/training data
        X2 = X.reshape(X.shape[0], 1, X.shape[1])
        cen = self.centroids.reshape(1, self.centroids.shape[0], X.shape[1])
        dists = np.sum((X2 - cen) ** 2, axis = 2)
        # dists have shape 10000, 10
        # This time unlike knn were we wanted the most votes, we want the smallest dists
        sorted_dists = np.argsort(dists, axis = 1)[:, 0]
        result = self.labels[sorted_dists]
        return result