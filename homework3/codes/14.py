import numpy as np
import regression
import data_preprocess

X_train, y_train = data_preprocess.preprocess_data("hw3_train.dat", x_0=1)
w_0 = np.ones(11)
regressor = regression.linear_regressor(w_0)
regressor.fit(X_train, y_train)
print(regressor.compute_mse(X_train, y_train))