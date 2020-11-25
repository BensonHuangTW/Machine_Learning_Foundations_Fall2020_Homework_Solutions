import numpy as np
import data_preprocess
import regression

# fetch data
X_train, y_train = data_preprocess.preprocess_data("hw3_train.dat", x_0=1)

# compute the E_in obtained by solving normal equation
w_0 = np.zeros(11)
regressor = regression.linear_regressor(w_0)
regressor.fit(X_train, y_train)
E_in_lin = regressor.compute_mse(X_train, y_train)
# use SGD to compute E_in
np.random.seed(42)

E_in_stop = 1.01 * E_in_lin
exp_times = 1000
train_iter_recorder = []
rand_seeds = np.random.randint(1, 10000, exp_times)


for i in range(exp_times):
    np.random.seed(rand_seeds[i])
    regressor.reset_weights(w_0)
    train_iter = regressor.sgd_fit(X_train, y_train, 1, 0.001, E_in_stop)
    train_iter_recorder.append(train_iter)
print(sum(train_iter_recorder) / exp_times)
