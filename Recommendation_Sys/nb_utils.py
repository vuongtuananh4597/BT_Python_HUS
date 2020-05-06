import numpy as np
from numba import njit


@njit
def _init(n_users, n_items, K):
    """
    """
    pu = np.random.normal(0, .1, (n_users, K))
    qi = np.random.normal(0, .1, (n_items, K))
    
    du = np.zeros(n_users)
    bi = np.zeros(n_items)

    return pu, qi, du, bi


@njit
def _update_use_bias(X, pu, qi, du, bi, global_mean, K, lr_rate, lambd):
    """
    """
    for i in range(X.shape[0]):
        user, item, rating = X[i, 0], X[i, 1], X[i, 2]
        pred_rating = global_mean + du[user] + bi[item]

        for k in range(K):
            pred_rating += pu[user, k] + qi[item, k]

        error = rating - pred_rating

        # Update biases
        du[user] += lr_rate * (error - lambd * du[user])
        bi[item] += lr_rate * (error - lambd * bi[user])

        # Update latent variables
        for k in range(K):
            puu = pu[user, k]
            qii = qi[user, k]

            pu[user, k] += lr_rate * (error*qii - lambd*puu)
            qi[item, k] += lr_rate * (error*puu - lambd*qii)

        return pu, qi, du, bi
    

@njit
def _update_no_bias(X, pu, qi, global_mean, K, lr_rate, lambd):
    """
    """
    for i in range(X.shape[0]):
        user, item, rating = X[i, 0], X[i, 1], X[i, 2]
        pred_rating = global_mean

        for k in range(K):
            pred_rating += pu[user, k] + qi[item, k]

        error = rating - pred_rating

        # Update latent variables
        for k in range(K):
            puu = pu[user, k]
            qii = qi[user, k]

            pu[user, k] += lr_rate * (error*qii - lambd*puu)
            qi[item, k] += lr_rate * (error*puu - lambd*qii)

        return pu, qi
    

@njit
def _compute_metric_use_bias(X_valid, pu, qi, du, bi, global_mean, K):
    """
    """
    residuals = []
    for i in range(X_valid.shape[0]):
        user, item, rating = int(X_valid[i, 0]), int(X_valid[i, 1]), X_valid[i, 2]
        pred = global_mean

        if user > -1:
            pred += du[user]

        if item > -1:
            pred += bi[item]

        if (user > -1) and (item > -1):
            for k in range(K):
                pred += pu[user, k] * qi[item, k]

        residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return (loss, rmse, mae)


@njit
def _compute_metric_no_bias(X_valid, pu, qi, global_mean, K):
    """
    """
    residuals = []
    for i in range(X_valid.shape[0]):
        user, item, rating = int(X_valid[i, 0]), int(X_valid[i, 1]), X_valid[i, 2]
        pred = global_mean

        if (user > -1) and (item > -1):
            for k in range(K):
                pred += pu[user, k] * qi[item, k]

        residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return (loss, rmse, mae)
