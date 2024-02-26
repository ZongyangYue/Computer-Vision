import numpy as np

class OneNearestNeighbor:
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

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        return np.array(y_pred)

# Example usage:
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 2], [3, 3]])
    y_test = np.array([0, 1])

    # Create and train KNN classifier
    knn = KNN()
    knn.fit(X_train, y_train)

    # Predict
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    # Calculate accuracies
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
