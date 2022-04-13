import time
import numpy as np

class ML:
    def __init__(self):
        pass
    
    def train(self, X, y):
        pass
        
    def predict(self, X):
        pass
    
    # Tests the training by test
    def test(self, X, y):
    
        # Makes prediction about data
        y_predict = self.predict(X)
        if y.shape != y_predict.shape:
            raise AssertionError("The shape of prediction is {} but it should be {}".format(y_predict.shape, y.shape))
        num_correct = np.count_nonzero(y == y_predict)
        
        # Ratios is a the percentage of corrects
        ratio = num_correct/y.shape[0]
        return num_correct, ratio, y_predict
        
        
class Timer:
    def __init__(self, model: ML, X_train, y_train, X_test, y_test, show_results = False):
        self.model = model
        self.model_name = model.__class__.__name__
        
        t1 = time.time()
        self.model.train(X_train, y_train)
        t2 = time.time()

        print("\n")

        print("===========================================================================================")
        
        print("Training benchmark for {b} with {l} datas = {a}s".format(a = round(t2 - t1, 5), b = self.model_name, l = y_train.shape[0]))
        
        t3 = time.time()
        nc, ratios, results = self.model.test(X_test, y_test)
        t4 = time.time()
        
        print("Testing benchmark for {b} with {l} datas = {a}s".format(a = round(t4 - t3, 5), b = self.model_name, l = y_test.shape[0]))
        print("Number of correct predictions: {a}, Accuracy: {r}%".format(a = nc, r = round(ratios * 100, 5)))
        if show_results:
            print("Results:")
            print(results)
        print("===========================================================================================")
        
        self.train_time = round(t2 - t1, 5)
        self.test_time = round(t4 - t3, 5)
        self.total = round(t2 - t1 + t4 - t3, 5)
        self.num_correct = nc
        self.accuracy = ratios