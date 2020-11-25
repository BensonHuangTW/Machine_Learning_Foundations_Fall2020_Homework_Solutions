import numpy as np
import regression
import data_preprocess

# first solve the linear regression problem to get w_lin
X_train, y_train = data_preprocess.preprocess_data("hw3_train.dat", x_0=1)
num_data = len(X_train)
w_lin = np.linalg.inv(X_train.T.dot(X_train))\
.dot(X_train.T).dot(y_train)

# hyperparameters
w_0 = w_lin
regressor = regression.logistic_regressor(w_0)
exp_time, iter_time, learning_rate = 1000, 500, 0.001
rand_seeds = np.random.randint(0, 1000, exp_time)

cross_entropy_records = []

for i in range(exp_time):
    np.random.seed(rand_seeds[i])
    regressor.fit(X_train, y_train, iter_time, learning_rate)
    cross_entropy_records.append(regressor.compute_cross_entropy \
    (X_train, y_train))
    regressor.reset_weights(w_0)

print(sum(cross_entropy_records) / exp_time)
