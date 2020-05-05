import numpy as np


def train_test_split(x, train_ratio, base=None):
    """Batch generator
    Params :
     - x : input matrix
     - ratio : split_ratio
    Outputs :
     - matrix (batch_size, :)
    """
    mask = np.random.permutation(len(x))
    x = x[mask]
    if base == 'item':
        ii = round(len(x)*train_ratio)
        x_train = x[:ii, :]
        x_test = x[ii:, :]
    elif base == 'user':
        ii = round(x.shape[1]*train_ratio)
        x_train = x[:, :ii]
        x_test = x[:, ii:]
    
    return x_train, x_test


def rmse(target, pred):
    """
    """
    return np.mean(np.square(target - pred)) ** .5