import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N * D where each row is an example. D is the dimension.
        Y is 1-dimension of size N 
        The training of KNN only simply memorize the training data and labels"""
        self.Xtrain = X
        self.ytrain = y
    
    def predict(self, X):
        """ X is N * D where each row is an example. We wish to predict labels for each of them
        return: y_pred, shape = (1, N)"""
        num_test = X.shape[0]

        # make sure the type is correct
        y_pred = np.zeros(num_test, dtype = self.ytrain.type)

        #loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i' th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtrain - X[i, :]), axis = 1)
            min_index = np.argmin(distances) #index with smallest distance -- nearest neighbor
            y_pred[i] = self.ytrain[min_index] #the label of the nearest example is assigned to the test example

        return y_pred