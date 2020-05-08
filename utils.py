import numpy as np


seed = 123
np.random.seed(seed)
def train_test_split(X, split_ratio, shuffle, output_type=None):
    """Train valid/test set
    Inputs:
     - X (DataFrame): input data
     - split_ratio (scalar)
     - shuffle (boolean)
    Outputs:
     - Tuple: train, valid & test set
    """
    if shuffle:
        idx = np.random.permutation(X.shape[0])
    else:
        idx = X.index.tolist()
    train_idx = int(len(idx) * split_ratio)
    X_train = X.iloc[idx[:train_idx]]
    X_valid = X.iloc[idx[train_idx:]].sample(frac=.5, random_state=seed)
    X_test = X.iloc[idx[train_idx:]].drop(X_valid.index.tolist())

    if output_type == 'np':
        return X_train.values, X_valid.values, X_test.values
    elif output_type == 'df':
        return X_train, X_valid, X_test