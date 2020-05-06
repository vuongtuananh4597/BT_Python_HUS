import numpy as np
import utils
import time


class fastSVD(object):
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
    def __init__(self, K, lambd=0.1, lr_rate=0.5, max_iter=50,):
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
        """
        """
        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))
        
        if self.early_stopping:
            list_losses = [10]
        
        if self.use_biases:
            pu, qi, du, bi = utils._init(n_users, n_items, self.K)
            for it in range(self.max_iter):
                st = time.time()
                print('Epoch {}/{}'.format(it+1, self.max_iter))
                
                pu, qi, du, bi = utils._update_use_bias(X, pu, qi, du, bi,
                                                        self.global_mean, self.K,
                                                        self.lr_rate,
                                                        self.lambd)
                if self.early_stopping:
                    valid_loss, valid_rmse, valid_mae = utils._compute_metric_use_bias(X_valid, pu, qi, du, bi,
                                                                                       self.global_mean, self.K)
                    
                    list_losses.append(valid_rmse)
                    print("Valid loss: {} --- Valid RMSE: {} --- Valid MAE: {}".format(valid_loss, valid_rmse, valid_mae))
                    # Break condition
                    if list_losses[-1] + 1e-5 > list_losses[-2]:
                        break
                elif self.verbose & ((it+1) % 10 == 0):
                    valid_loss, valid_rmse, valid_mae = utils._compute_metric_use_bias(X_valid, pu, qi, du, bi,
                                                                                       self.global_mean, self.K)
                    print("Valid loss: {} --- Valid RMSE: {} --- Valid MAE: {}".format(valid_loss, valid_rmse, valid_mae))
                    
            self.pu = pu
            self.qi = qi
            self.du = du
            self.bi = bi
        else:
            pu, qi, _, _ = utils._init(n_users, n_items, self.K)
            for it in range(self.max_iter):
                st = time.time()
                print('Epoch {}/{}'.format(it+1, self.max_iter))
                
                pu, qi = utils._update_no_bias(X, pu, qi, self.global_mean,
                                                self.K, self.lr_rate, self.lambd)
                if self.early_stopping:
                    valid_loss, valid_rmse, valid_mae = utils._compute_metric_no_bias(X_valid, pu,
                                                                                       self.global_mean, self.K)
                    
                    list_losses.append(valid_rmse)
                    print("Valid loss: {} --- Valid RMSE: {} --- Valid MAE: {}".format(valid_loss, valid_rmse, valid_mae))
                    # Break condition
                    if list_losses[-1] + 1e-5 > list_losses[-2]:
                        break
                elif self.verbose & ((it+1) % 10 == 0):
                    valid_loss, valid_rmse, valid_mae = utils._compute_metric_no_bias(X_valid, pu, qi,
                                                                                       self.global_mean, self.K)
                    print("Valid loss: {} --- Valid RMSE: {} --- Valid MAE: {}".format(valid_loss, valid_rmse, valid_mae))
                    
            self.pu = pu
            self.qi = qi


    def fit(self, X, X_valid, early_stopping, verbose, use_biases):
        """
        """
        self.verbose = verbose
        self.use_biases = use_biases
        self.early_stopping = early_stopping
        print("Load data & Preprocessing !")
        
        X = self._preprocess(X, train=True)
        X_valid = self._preprocess(X_valid, train=False)
        
        self.global_mean = np.mean(X[:, 2])
        self._run(X, X_valid)
       

    def predict_given_id(self, userID, movieID):
        """Predict rating value given userID and movieID
        """
        pred = self.global_mean
        if userID in self.user_dict:
            userid = self.user_dict[userID]
            pred += self.du[userid]
        
        if movieID in self.item_dict:
            movieid = self.item_dict[movieID]
            pred += self.bi[movieid]
        
        pred += np.dot(self.pu[userid], self.qi[movieid])
        
        return max(0, min(5, pred))


    def predict(self, X):
        """Predict rating value given matrix
        """
        pred = []
        for uid, iid in zip(X['userID'], X['movieID']):
            pred.append(self.predict_given_id(uid, iid))
        
        return pred