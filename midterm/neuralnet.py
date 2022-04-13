from ml import ML, Timer
import numpy as np

class MultiLayerPerceptron(ML):
    # Net dimensions: (neuron per hidden layer, number of hidden layers)
    # Data dimensions: (number of inputs, number of outputs)
    # max number of epochs
    def __init__(self, data_dimensions : tuple, net_dimensions : tuple = (128, 2), max_epoch = 1):
        # Randomly generate all the weights
        self.wij = np.random.rand(data_dimensions[0], net_dimensions[0])
        self.wjj = np.random.rand(net_dimensions[1] - 1, net_dimensions[0], net_dimensions[0])
        self.wjk = np.random.rand(net_dimensions[0], data_dimensions[1])
        self.tj = np.random.rand(*net_dimensions)
        self.tk = np.random.rand(data_dimensions[1])

        print("Shapes: wij: {}, wjj: {}, wjk: {}, tj: {}, tk: {}".format(self.wij.shape, self.wjj.shape, self.wjk.shape, self.tj.shape, self.tk.shape))
        
        self.max_epoch = max_epoch
        
        # My slow approach of adjusting the learning rate on the fly
        # Learning rate is calculated as initial * lr_coeff
        # If we think learning rate is too fast we decrease lr_coeff and then try again
        self.initial_learning_rate = 1 / (net_dimensions[0] * net_dimensions[1])
        self.learning_rate = self.initial_learning_rate
        self.lr_coefficient = 1
        
        self.layers = net_dimensions[1]
        self.num_outputs = data_dimensions[1]
        
    # Decreases learning rate
    def update_learning_rate(self, nudge = 1.1):
        self.lr_coefficient *= nudge
        self.learning_rate = self.initial_learning_rate ** self.lr_coefficient
        print(self.learning_rate)
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def dsigmoid(self, z):
        return z * (1-z)
        
        
    def fwprop(self, X):
        # First hidden layer
        output_j1 = np.dot(X, self.wij)
        output_j1 = self.sigmoid(output_j1 + self.tj[0])
        
        self.output_jn = np.zeros((self.layers, self.wjj.shape[-1]))
        # print(self.output_jn.shape)
        # print(output_j1.shape)
        self.output_jn[0] = output_j1
        
        # output_jn = a 2d aray wiere its i-th row storing the i-th hidden layer output
        # Do all subsequent layers
        for i in range(1, self.layers):
            # dot the last output and the weights to obtain something with shape (128,)
            sum = np.dot(self.output_jn[i - 1], self.wjj[i - 1])
            self.output_jn[i] = self.sigmoid(sum + self.tj[i])
        
        # Final layer
        # We wont use soft max because it is not taught yet and we don't want to get caught up in details
        output_k = np.dot(self.output_jn[-1], self.wjk)
        output_k = self.sigmoid(output_k + self.tk)
        
        self.output_k = output_k
        return output_k
        
    def backprop(self, X, y):
        # Keep track of the total nudge so we can determine if it is increasing later
        total_nudge = 0
        
        # First (ahem I mean last) layer
        # d_k is the batched averaged delta_ks
        # shape: (number of outputs, )
        delta_k = -(self.output_k - y) * self.dsigmoid(self.output_k)
        
        # Update weights and biases for last layer
        self.tk = self.tk - self.learning_rate * delta_k
        O_j = self.output_jn[-1].reshape(-1, 1)
        d_k = delta_k.reshape(1, -1)
        self.wjk = self.wjk - self.learning_rate * O_j * d_k
        
        # We will only consider delta k since if delta k is small the overall change will be small
        total_nudge += self.learning_rate * np.sum(np.abs(delta_k))
        
        # Store the "next layer" weights and deltas for updating
        # Shape = (last layer # neurons)
        sums = np.dot(self.wjk, delta_k)
        
        # Decrementally back prop
        i = self.layers - 1
        while i > 0:
            # First calculate the sum d_k w_jk part
            # Shape = (# neurons in hidden layer)
            delta_j = sums * self.dsigmoid(self.output_jn[i])
            
            # Update weights and biases
            self.tj = self.tj - self.learning_rate * delta_j
            
            O_j = self.output_jn[i - 1].reshape(-1, 1)
            d_j = delta_j.reshape(1, -1)
            self.wjj[i-1] = self.wjj[i-1] - self.learning_rate * O_j * d_j
            
            # Store sums for next layer use
            sums = np.dot(self.wjj[i-1], delta_j)
            total_nudge += self.learning_rate * np.sum(np.abs(delta_j))
            i -= 1
            
        # Update first layer
        delta_j0 = sums * self.dsigmoid(self.output_jn[0])
        
        # Update weights and biases
        self.tj = self.tj - self.learning_rate * delta_j0
        
        O_j = X.reshape(-1, 1)
        d_j = delta_j0.reshape(1, -1)
        self.wij = self.wij - self.learning_rate * O_j * d_j
        total_nudge += self.learning_rate * np.sum(np.abs(delta_j0))
        
        return total_nudge

    def train(self, X, Y):
        # Preprocess Y data
        self.unique_outputs = np.unique(Y)
        assert len(self.unique_outputs) == self.num_outputs

        ind = self.unique_outputs + np.zeros((Y.shape[0], self.num_outputs))
        t = Y.reshape(-1, 1) == ind
        Y_train = np.array(t, dtype = int)

        run = 0
        converge = False
        last_total_nudges = np.full((5,), 99999, dtype = np.float64)
        while run < self.max_epoch and not converge:
            total_nudge = 0
            for i in range(len(X)):
                x = X[i]
                y = Y_train[i]
                self.fwprop(x)
                total_nudge += self.backprop(x, y)
                
            # Store the last total nudges
            last_total_nudges[0 : -1] = last_total_nudges[1 :]
            last_total_nudges[-1] = total_nudge
            
            # print(last_total_nudges)
            
            # If the last few total nudges are ascending, that means we are exploding lol
            # The perceptron realizes it is too dangerous to be kept alive and decreases the learning rate
            if total_nudge > 100 or isIncreasing(last_total_nudges):
                self.update_learning_rate()
                return self.train(X, Y)
            
            # We want the average nudge of each variable to be less than 0.001
            threshold = min(0.001 * len(X) * (self.wjj.size + self.wij.size + self.wjk.size + self.tj.size + self.tk.size), 0.001)
            print("Total nudge = {}, threshold = {}, run = {}".format(total_nudge, threshold, run))
            if total_nudge < threshold:
                converge = True

            run += 1
    
    def predict(self, X):
        # First hidden layer
        output_j1 = np.dot(X, self.wij)
        output_j1 = self.sigmoid(output_j1 + self.tj[0])
        output_jn = np.zeros((len(X), self.layers, self.wjj.shape[-1]))
        output_jn[:, 0, :] = output_j1
        for i in range(1, self.layers):
            sum = np.dot(output_jn[:, i - 1, :], self.wjj[i - 1])
            output_jn[:, i, :] = self.sigmoid(sum + self.tj[i])
        output_k = np.dot(output_jn[:, -1, :], self.wjk)
        output_k = self.sigmoid(output_k + self.tk)
        # print(output_k)

        # Now output k is the probability array thing
        sorted_prob = np.argsort(output_k, axis = 1)
        best_guess = self.unique_outputs[sorted_prob[:, -1]]
        # print(sorted_prob)
        return best_guess
        
    def export_data(self):
        np.save("wij", self.wij)
        np.save("wjk", self.wjk)
        np.save("wjj", self.wjj)
        np.save("tj", self.tj)
        np.save("tk", self.tk)
    
    def import_data(self, wij, wjk, wjj, tj, tk):
        self.wij = np.load(wij)
        self.wjk = np.load(wjk)
        self.wjj = np.load(wjj)
        self.tj = np.load(tj)
        self.tk = np.load(tk)




def isIncreasing(lis):
    lista = np.array(lis[:-1])
    listb = np.array(lis[1:])
    # print("Lists: ")
    # print(lista, listb)
    incr = np.count_nonzero(listb > lista + 1e-3)
    # print("Num incr = %s" % incr)
    return incr >= len(lista) - 1