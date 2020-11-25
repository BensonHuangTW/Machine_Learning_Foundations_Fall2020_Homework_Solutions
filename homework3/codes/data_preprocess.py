import numpy as np
from numpy import genfromtxt

def preprocess_data(file_name, x_0, scaling_factor=1):
    # preprocess the data
    raw_data = genfromtxt(file_name)
    X = raw_data[:, :10].copy()
    y = raw_data[:, 10].copy()
    # add x_0 into input
    X = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1) * x_0 , X], axis=1) / scaling_factor
    return X, y.reshape((len(y), 1))

def poly_transform_data(file_name, order):
    # preprocess the data
    raw_data = genfromtxt(file_name)
    X = raw_data[:, :10].copy()
    y = raw_data[:, 10].copy()

    # add polynomial features to data
    new_X = X
    high_order_feature = X
    for i in range(order - 1):
        high_order_feature = high_order_feature * X
        new_X = np.c_[new_X, high_order_feature]
    new_X = np.c_[np.ones(X.shape[0]).reshape(-1, 1), new_X]
    return new_X, y.reshape((len(y), 1))