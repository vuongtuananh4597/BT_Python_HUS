import numpy as np
from numba import njit, roc


@njit
def _init(n_users, n_items, K):
    """Initialization latent factors
    """
    pu = np.random.normal(0, .1, (n_users, K))
    qi = np.random.normal(0, .1, (n_items, K))

    return pu, qi


@njit
def _update(X, pu, qi, user_num, user_denom, item_num, item_denom , K, lambd_pu, lambd_qi):
    """Update function for funkSVD (no biases)
    """
    for i in range(X.shape[0]):
        user, item, rating = X[i, 0], X[i, 1], X[i, 2]
        pred_rating = 0
        for k in range(K):
            pred_rating += pu[user, k] * qi[item, k]
            
        err = rating - pred_rating

        # Compute numerators and denominators
        for k in range(K):
            user_num[user, k] += qi[item, k] * rating
            user_denom[user, k] += qi[item, k] * pred_rating
            item_num[item, k] += pu[user, k] * rating
            item_denom[item, k] += qi[item, k] *pred_rating
        
    # Update user factors (pu)
    for user in np.unique(X[:, 0]):
        n_ratings = len(X[X[:, 0] == user])
        for k in range(K):
            user_denom[user, k] += n_ratings * lambd_pu * pu[user, k]
            pu[user, k] *= user_num[user, k] / user_denom[user, k]

    # Update item factors (qi)    
    for item in np.unique(X[:, 1]):
        n_ratings = len(X[X[:, 1] == item])
        for k in range(K):
            item_denom[item, k] += n_ratings * lambd_qi * qi[item, k]
            qi[item, k] *= item_num[item, k] / item_denom[item, k]

    return pu, qi
    

@njit
def _compute_metric(X_valid, pu, qi, K):
    """Compute metrics for no bias ver.
    """
    residuals = []
    for i in range(X_valid.shape[0]):
        user, item, rating = int(X_valid[i, 0]), int(X_valid[i, 1]), X_valid[i, 2]
        pred_rating = 0
        if (user > -1) and (item > -1):
            for k in range(K):
                pred_rating += pu[user, k] * qi[item, k]

        residuals.append(rating - pred_rating)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return (loss, rmse, mae)


class NMF(object):
    """Matrix factorization based on Non-negative Matrix Factorization
    This algo is very similar to funkSVD. It is in non-regularized form and for dense matrices
    Solver used in this implementation is Multiplicative update rules (Gonna update more !!)
    
    Params:
     - K (int): number of latent factors
     - lambd_pu (float): regularization param for users
     - lambd_qi (float): regularization param for items
     - max_iter (int): number of max iteration(epochs)
     - pu (array): initialize users latent factor matrix
     - qi (array): initialize items latent factor matrix
     - early_stopping (boolean): early stopping technique
     
    """ 
    def __init__(self, K, lambd_pu=0.07, lambd_qi=0.07, max_iter=50):
        self.K = K
        self.lambd_pu = lambd_pu
        self.lambd_qi = lambd_qi
        self.max_iter = max_iter


    def _preprocess(self, X, train=True):
        """Prepprocessing data from DataFrame to numpy array
        // [userID, movieID, rating]
        """
        X = X.copy()
        
        if train:
            user_idx = X['userID'].unique().tolist()
            item_idx = X['movieID'].unique().tolist()
            
            self.user_dict = dict(zip(user_idx, [i for i in range(len(user_idx))]))
            self.item_dict = dict(zip(item_idx, [i for i in range(len(item_idx))]))
        
        X['userID'] = X['userID'].map(self.user_dict)
        X['movieID'] = X['movieID'].map(self.item_dict)
        
        # set unknown with -1 (valid)
        X.fillna(-1, inplace=True)
        
        X['userID'] = X['userID'].astype(np.int32)
        X['movieID'] = X['movieID'].astype(np.int32)
        
        X = X[['userID', 'movieID', 'rating']].values
        
        return X


    def _run(self, X, X_valid):
        """Helper function for fit function
        """
        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))
        pu, qi = _init(n_users, n_items, self.K)

        for it in range(self.max_iter):
            print('Epoch {}/{}'.format(it+1, self.max_iter))
            
            user_num = np.zeros((n_users, self.K))
            user_denom = np.zeros((n_users, self.K))
            item_num = np.zeros((n_items, self.K))
            item_denom = np.zeros((n_items, self.K))

            pu, qi = _update(X, pu, qi, user_num, user_denom, item_num, item_denom, self.K, self.lambd_pu, self.lambd_qi)
            if self.early_stopping:
                valid_loss, valid_rmse, valid_mae = _compute_metric(X_valid, pu, qi, self.K)
                
                print("Valid loss: {} --- Valid RMSE: {} --- Valid MAE: {}".format(valid_loss, valid_rmse, valid_mae))
                # Break condition
                if self.list_losses and (valid_rmse + 1e-4 > list_losses[-1]):
                    break

                self.list_losses.append(valid_rmse)
            elif self.verbose and ((it+1) % 10 == 0):
                valid_loss, valid_rmse, valid_mae = _compute_metric(X_valid, pu, qi, self.K)
                print("Valid loss: {} --- Valid RMSE: {} --- Valid MAE: {}".format(valid_loss, valid_rmse, valid_mae))
                self.list_losses.append(valid_rmse)

            self.pu = pu
            self.qi = qi


    def fit(self, X, X_valid, early_stopping, verbose):
        """Function of training phase
        """
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.list_losses = []
        print("Load data & Preprocessing !")
        
        X = self._preprocess(X, train=True)
        X_valid = self._preprocess(X_valid, train=False)
        
        self._run(X, X_valid)
       

    def predict_given_id(self, userID, movieID):
        """Predict rating value given userID and movieID
        """
        pred += np.dot(self.pu[userid], self.qi[movieid])
        
        return max(0, min(5, pred))


    def predict(self, X):
        """Predict rating value given matrix
        """
        pred = []
        for uid, iid in zip(X['userID'], X['movieID']):
            pred.append(self.predict_given_id(uid, iid))
        
        return pred