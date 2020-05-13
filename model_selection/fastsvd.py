import numpy as np
from numba import njit, roc


@njit
def _init(n_users, n_items, K):
    """Initialization matrix/vector
    """
    pu = np.random.normal(0, .1, (n_users, K))
    qi = np.random.normal(0, .1, (n_items, K))
    
    du = np.zeros(n_users)
    bi = np.zeros(n_items)

    return pu, qi, du, bi


@njit
def _update(X, pu, qi, du, bi, global_mean, K, lr_rate, lambd, use_biased):
    """Update function for funkSVD (use biases)
    """
    for i in range(X.shape[0]):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred_rating = global_mean + du[user] + bi[item]

        for k in range(K):
            pred_rating += pu[user, k] * qi[item, k]

        error = rating - pred_rating

        # Update biases
        if use_biased:
            du[user] += lr_rate * (error - lambd * du[user])
            bi[item] += lr_rate * (error - lambd * bi[item])

        # Update latent variables
        for k in range(K):
            puf = pu[user, k]
            qif = qi[item, k]

            pu[user, k] += lr_rate * (error*qif - lambd*puf)
            qi[item, k] += lr_rate * (error*puf - lambd*qif)

    return pu, qi, du, bi
    

@njit
def _compute_metric(X_valid, pu, qi, du, bi, global_mean, K):
    """Compute some metrics (loss, rmse, mae) for evaluation
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


class fastSVD():
    """Matrix factorization based on funkSVD (fast version)
    
    Params:
     - K (int): number of latent factors
     - lambd (int): regularization param
     - lr_rate (float): learning rate
     - max_iter (int): number of max iteration(epochs)
     - pu (array): initialize users latent factor matrix
     - qi (array): initialize items latent factor matrix
     - du (array): users biases vector
     - bi (array): items biases vector
     - early_stopping (boolean): early stopping technique
     
    """ 
    def __init__(self, K=15, lambd=0.1, lr_rate=0.5, max_iter=50,):
        self.K = K
        self.lambd = lambd
        self.lr_rate = lr_rate
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
        
        pu, qi, du, bi = _init(n_users, n_items, self.K)
        for it in range(self.max_iter):
            print('Epoch {}/{}'.format(it+1, self.max_iter))

            pu, qi, du, bi = _update(X, pu, qi, du, bi,
                                     self.global_mean, self.K,
                                     self.lr_rate,
                                     self.lambd, self.use_biased)

            if self.early_stopping:
                valid_loss, valid_rmse, valid_mae = _compute_metric(X_valid, pu, qi, du, bi,
                                                                    self.global_mean, self.K)

                print("Valid loss: {:.5f} --- Valid RMSE: {:.5f} --- Valid MAE: {:.5f}".format(valid_loss, valid_rmse, valid_mae))
                # Break condition
                if self.list_losses and (valid_rmse + 1e-4 > self.list_losses[-1]):
                    break

                self.list_losses.append(valid_rmse)
            elif self.verbose & ((it+1) % 10 == 0):
                valid_loss, valid_rmse, valid_mae = _compute_metric(X_valid, pu, qi, du, bi,
                                                                    self.global_mean, self.K)
                print("Valid loss: {:.5f} --- Valid RMSE: {:.5f} --- Valid MAE: {:.5f}".format(valid_loss, valid_rmse, valid_mae))
                self.list_losses.append(valid_rmse)

        self.pu = pu
        self.qi = qi
        self.du = du
        self.bi = bi


    def fit(self, X, X_valid, early_stopping, verbose, use_biased):
        """Function of training phase
        """
        self.verbose = verbose
        self.use_biased = use_biased
        self.early_stopping = early_stopping
        self.list_losses = []
        print("Load data & Preprocessing !")
        
        X = self._preprocess(X, train=True)
        X_valid = self._preprocess(X_valid, train=False)
        
        self.global_mean = np.mean(X[:, 2])
        self._run(X, X_valid)
    
        
    def predict_given_id(self, userID, movieID):
        """Predict rating value given userID and movieID
        """
        user_known, item_known = False, False
        pred = self.global_mean
        if userID in self.user_dict:
            user_known = True
            userid = self.user_dict[userID]
            pred += self.du[userid]

        if movieID in self.item_dict:
            item_known = True
            movieid = self.item_dict[movieID]
            pred += self.bi[movieid]
        
        if user_known & item_known:
            pred += np.dot(self.pu[userid], self.qi[movieid])
        
        return max(0, min(5, pred))
   

    def predict_for_user(self, userID):
        """Predict rating value for all items given userID
        """
        pred = {}
        for movieID in list(self.item_dict.keys()):
            pred[movieID] = self.predict_given_id(userID, movieID)
            
        return pred


    def predict(self, X):
        """Predict rating value given matrix
        """
        pred = []
        for uid, iid in zip(X['userID'], X['movieID']):
            pred.append(self.predict_given_id(uid, iid))
        
        return pred
    

    def evaluate(self, X_valid):
        """
        """
        pred_rating = np.array(self.predict(X_valid.iloc[:, :2]))
        true_rating = X_valid.iloc[:, 2]
        
        return np.sqrt(np.mean((true_rating - pred_rating)**2))
                               
        