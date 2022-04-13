import numpy as np

class Perceptron:
    # We need f, w, theta, number of variables k, learning rate n
    def __init__(self, f=None, k=2, w=None, theta=None, n = 0.02, magicalConstant = -2.9):
        # Handle the unhandled cases :)
        if f == None:
            # Use f(x) = x for the sake of simplicity
            def f(x):
                return x
        if type(w) == type(None):
            w = np.random.rand(k)
        if theta == None:
            theta = np.random.rand()

        self.w = w
        self.f = np.vectorize(f)

        # Sanity check
        assert k == len(w)

        self.k = len(self.w)
        self.theta = theta
        self.orig_n = n
        self.adjustLearningRate(magicalConstant)
    
    # Haha adjusting learning rate on the fly
    def adjustLearningRate(self, magicalConstant):
         self.n = self.orig_n * self.k ** ( magicalConstant )
         self.magicalConstant = magicalConstant
    

    # Evaluates f at theta + w * x and returns O
    def evaluate(self, x):
        input = np.dot(x, self.w) + self.theta
        return self.f(input)
    

    def learn(self, trainX, trainY, maxIter = 1000):
        iter = 0
        while True:
            iter += 1
            
            if iter > maxIter:
                print("Reached max iteration count! Returning")
                return

            # evaluates X at every row, then subtract Y to get data
            results = self.evaluate(trainX)
            errs =  trainY - results
            totalerr = np.sum(errs)

            # print("Total Error: " + str(totalerr))
            
            # If error small enough then training successful, return
            if np.abs(totalerr) <= 0.5:
                return
            
            # I found a perceptron inside a perceptron works wonders to adjust the learning rate on the fly
            elif self.n * np.abs(totalerr) >= self.k:
                # number of iterations sort of represent the speed it diverges - empirically the constant seems to be between 2 and 3
                # If it diverges quickly then we want to nudge the learning rate more
                newConst = self.magicalConstant - 2.71828 ** (-iter)
                self.adjustLearningRate(newConst)
                # print("Adjusting learning rate. New magical constant = " + str(newConst))
                return self.learn(trainX, trainY, maxIter)
            else:
                # dw = n * (T - O) * x
                # dt = n * (T - O)
                # Since x is a 2d matrix, err is a 1d vector corresponding to the k-th row's error, We can use linear algebra magic to do this in one step
                err = self.n * np.dot(errs, trainX)
                
                # Now adjust
                self.w += err
                self.theta += self.n * totalerr

    def predict(self, x):
        return self.evaluate(x)