import numpy as np


class SVD(object):
    """Matrix factorization based on funkSVD (numpy ver.)

    Params:
     - K : number of latent factors
     - lambd : regularization param
     - lr_rate : learning rate
     - max_iter: number of max iterations
     - verbose (boolean): traceback losses (for debug)
     - user_based (boolean): normalization by user column or item column
     - use_biased (boolean): use biases or not
    """ 
    def __init__(self, train, valid, K, lambd=0.1, lr_rate=0.5, max_iter=1000, Xinit=None,
                 Winit=None, verbose=True, user_based=True, use_biased=None):
        
        self.raw = valid  # for evaluation
        self.data = train  # for training
        self.raw = self.raw.values
        self.data = self.data.values

        self.raw[:, :2] -= 1
        self.data[:, :2] -= 1
        
        self.n_users = int(np.max(self.data[:, 0])) + 1 
        self.n_items = int(np.max(self.data[:, 1])) + 1
        self.n_ratings = len(self.data)
        self.global_mean = np.mean(self.data[:, 2])

        self.K = K
        self.lambd = lambd
        self.lr_rate = lr_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.user_based = user_based
        self.use_biased = use_biased
        self.train_loss = []  # to keep track loss while training
        self.train_rmse = []

        if Xinit is None: 
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = Xinit 
        
        if Winit is None: 
            self.W = np.random.randn(K, self.n_users)
        else: 
            self.W = Winit
        
        # biases for item, user
        if use_biased:
            self.b_i = np.random.randn(self.n_items)
            self.d_u = np.random.randn(self.n_users)


    def _normalization(self):
        """Normalize (user or item) before training
        """


        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        else:
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = self.data[:, user_col] 
        self.keep_mean = np.zeros((n_objects,))
        for n in range(n_objects):
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.data[ids, 2]
            m = np.nanmean(ratings) 
            if np.isnan(m):
                m = 0
            self.keep_mean[n] = m

            self.data[ids, 2] = ratings - m
            

    def loss(self):
        """Calculate loss function
        """
        L = 0 
        if self.use_biased:
            for i in range(self.n_ratings):
                # user, item, rating
                n, m, rate = int(self.data[i, 0]), int(self.data[i, 1]), self.data[i, 2]
                L += 0.5*(self.X[m, :].dot(self.W[:, n]) + \
                          self.b_i[m] + self.d_u[n] + self.global_mean - rate)**2
                L /= self.n_ratings
                L += 0.5*self.lambd*(np.linalg.norm(self.X, 'fro') + \
                                     np.linalg.norm(self.W, 'fro') + \
                                     np.linalg.norm(self.b_i) + \
                                     np.linalg.norm(self.d_u)
                                    )
        else:
            for i in range(self.n_ratings):
                n, m, rate = int(self.data[i, 0]), int(self.data[i, 1]), self.data[i, 2]
                L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2
                L /= self.n_ratings
                L += 0.5*self.lambd*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))

        return L 
    
    def get_items_rated_by_user(self, user_id):
        """
        """
        ids = np.where(self.data[:,0] == user_id)[0] 
        item_ids = self.data[ids, 1].astype(np.int32)
        ratings = self.data[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        """
        ids = np.where(self.data[:,1] == item_id)[0] 
        user_ids = self.data[ids, 0].astype(np.int32)
        ratings = self.data[ids, 2]
        return (user_ids, ratings)


    def updateX(self):
        """Backprop and update X using gradient descent
        """
        if self.use_biased:
            for m in range(self.n_items):
                user_ids, ratings = self.get_users_who_rate_item(m)

                Wm = self.W[:, user_ids]
                dm = self.d_u[user_ids]
                Xm = self.X[m, :]
            
                error = Xm.dot(Wm) + dm + self.b_i[m] + self.global_mean - ratings 

                grad_Xm = error.dot(Wm.T)/self.n_ratings + self.lambd * Xm
                grad_dm = np.sum(error)/self.n_ratings + self.lambd * self.b_i[m]
                self.X[m, :] -= self.lr_rate * grad_Xm.reshape((self.K,))
                self.b_i[m] -= self.lr_rate * grad_dm
        else:
            for m in range(self.n_items):
                user_ids, ratings = self.get_users_who_rate_item(m)

                Wm = self.W[:, user_ids]
                grad_Xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
                                               self.lambd*self.X[m, :]
                self.X[m, :] -= self.lr_rate*grad_Xm.reshape((self.K,))


    def updateW(self):
        """Backprop and update W using gradient descent
        """
        if self.use_biased:
            for n in range(self.n_users):
                item_ids, ratings = self.get_items_rated_by_user(n)
                Xn = self.X[item_ids, :]
                bn = self.b_i[item_ids]
                Wn = self.W[:, n]

                error = Xn.dot(Wn) + bn + self.d_u[n] + self.global_mean - ratings

                grad_Wn = Xn.T.dot(error) / self.n_ratings + self.lambd * Wn
                grad_bn = np.sum(error) / self.n_ratings + self.lambd * self.d_u[n]
                self.W[:, n] -= self.lr_rate * grad_Wn.reshape((self.K,))
                self.d_u[n] -= self.lr_rate * grad_bn
        else:
            for n in range(self.n_users):
                item_ids, ratings = self.get_items_rated_by_user(n)
                Xn = self.X[item_ids, :]

                grad_Wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
                            self.lambd * self.W[:, n]
                self.W[:, n] -= self.lr_rate*grad_Wn.reshape((self.K,))

                
    def fit(self):
        """Training phase
        """
        self._normalization()

        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if self.verbose & ((it+1) % 10 == 0):
                rmse_train = self.evaluate(self.raw)
                if self.train_rmse and (abs(self.train_rmse[-1] - rmse_train) < 1e-5):
                    print('iter =', it + 1, ', loss =', self.loss(), ', RMSE train before =', self.train_rmse[-1], ',RMSE train after =', rmse_train)
                    print("Stop")
                    break
                self.train_loss.append(self.loss())
                self.train_rmse.append(rmse_train)
                print('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)


    def predict(self, u, i):
        """Prediction given user_id and item_id
        """
        if self.user_based:
            bias = self.keep_mean[u]
        else:
            bias = self.keep_mean[i]
        if self.use_biased:
            pred = self.X[i, :].dot(self.W[:, u]) + self.b_i[i] + self.d_u[u] + bias
        else:
            pred = self.X[i, :].dot(self.W[:, u]) + bias
        
        return max(0, min(5, pred))
        
    
    def predict_for_user(self, user_id):
        """Prediction given user_id
        """
        ids = np.where(self.data[:, 0] == (user_id - 1))[0]
        # items_rated_by_u = self.data[ids, 1].tolist()              
        
        y_pred = self.X.dot(self.W[:, user_id])
        pred = {}
        for i in range(self.n_items):
            pred[i] = y_pred[i]
    
        return pred
    
    def evaluate(self, test_set):
        """Evaluation (RMSE/ MSE metric)
        """
        n_tests = test_set.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = self.predict(test_set[n, 0], test_set[n, 1])
            SE += (pred - test_set[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)

        return RMSE