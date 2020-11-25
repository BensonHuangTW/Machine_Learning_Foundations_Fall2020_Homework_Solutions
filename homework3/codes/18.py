import numpy as np
import data_preprocess
import regression

# fetch data
X_train, y_train = data_preprocess.preprocess_data("hw3_train.dat", x_0=1)
X_test, y_test = data_preprocess.preprocess_data("hw3_test.dat", x_0=1)

# initialize regressor 
w_0 = np.zeros(11)
regressor = regression.linear_regressor(w_0)

# training
regressor.fit(X_train, y_train)
E_in = 1 - regressor.compute_accuracy(X_train, y_train)
E_out = 1 - regressor.compute_accuracy(X_test, y_test)
print(np.abs(E_in - E_out))