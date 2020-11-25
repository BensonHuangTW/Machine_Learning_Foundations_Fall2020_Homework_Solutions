import numpy as np
import data_preprocess
import regression

# fetch data
order = 3
X_train, y_train = data_preprocess.poly_transform_data("hw3_train.dat", order=order)
X_test, y_test = data_preprocess.poly_transform_data("hw3_test.dat", order=order)

# initialize regressor and hyperparameters
w_0 = np.zeros(1 + 10 * order)
regressor = regression.linear_regressor(w_0)

# training
regressor.fit(X_train, y_train)
E_in = 1 - regressor.compute_accuracy(X_train, y_train)
E_out = 1 - regressor.compute_accuracy(X_test, y_test)
print(np.abs(E_in - E_out))