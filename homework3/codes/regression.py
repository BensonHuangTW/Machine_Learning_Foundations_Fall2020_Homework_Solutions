import numpy as np

class linear_regressor():
    def __init__(self, w):
        self.w = w.reshape((len(w), 1))
        return

    def reset_weights(self, w):
        self.w = w.reshape((len(w), 1))
        return

    def compute_mse(self, X, y):
        """
        This funciton compute the mean square error of the model.
        Please make sure that the shapes of X and y are (n,m) and (n,1)
        """
        return np.linalg.norm((X @ self.w) - y) ** 2 / len(y)

    def compute_accuracy(self, X, y):
        """
        This function compute the prediction acccuracy of the model.
        Please make sure that the shapes of X and y are (n,m) and (n,1)
        """
        result = np.sign(X @ self.w) * y
        return np.count_nonzero(result == 1) / len(y)

    def fit(self, X, y):
        """
        This function does linear regression by solving normal equation
        """
        self.w = np.linalg.inv(X.T.dot(X))\
            .dot(X.T).dot(y)
        return
    
    def sgd_fit(self, X, y, iter_time, lr, E_in_stop=None):
        """
        This funciton optimizes mean square error by SGD, and return
        the number of iterations.
        If E_in_stop == None, SGD will stop after iter_time iteration.
        If E_in_stop == E, SGD will will not stop until E_in <= E.
        """
        num_data = len(X)
        def update_weight():
            # use sgd to update weights for one time
            random_index = np.random.randint(num_data)
            x_t = X[random_index:random_index+1, :]
            y_t = y[random_index:random_index+1, :]
            grad = 2 * x_t.T @ (x_t.dot(self.w) - y_t)
            self.w = self.w - lr * grad
            return
        if E_in_stop != None:
            num_iter = 0
            while self.compute_mse(X, y) > E_in_stop:
                update_weight()
                num_iter += 1
            return num_iter
        else:
            for _ in range(iter_time):
                update_weight()
            return iter_time            

class logistic_regressor():
    def __init__(self, w):
        self.w = w.reshape((len(w), 1))
        return

    def reset_weights(self, w):
        self.w = w.reshape((len(w), 1))
        return

    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_accuracy(self, X, y):
        """
        This function compute the 0/1 error of the model.
        Please make sure that the shapes of X and y are (n,m) and (n,1)
        """
        result = np.sign(X @ self.w) * y
        return np.count_nonzero(result == 1) / len(y)


    def compute_cross_entropy(self, X: np.array, y: np.array):
        """
        This function compute the crossentropy for binary classification.
        Please make sure that the shapes of X and y are (n,m) and (n,1)
        """
        z = -(X @ self.w) * y
        return np.sum(np.log(1 + np.exp(z))) / len(y)

    def compute_CE_grad(self, X: np.array, y:np.array):
        """
        This function compute the gradient of crossentropy wrt weights.
        Please make sure that the shapes of X and y are (n,m) and (n,1)
        """
        z = -(X @ self.w) * y
        return - X.T @ (self.logistic(z) * y) / len(y)

    def fit(self, X: np.array, y: np.array, iter_time, lr):
        for _ in range(iter_time):
            random_index = np.random.randint(len(y))
            x_t = X[random_index:random_index+1, :]
            y_t = y[random_index:random_index+1, :]
            grad = self.compute_CE_grad(x_t, y_t)
            self.w = self.w - lr * grad
        return