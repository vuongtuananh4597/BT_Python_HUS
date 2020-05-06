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
    def __init__(self, K, lambd=0.1, lr_rate=0.5, max_iter=1000,
                 verbose=True, user_based=True, use_biased=None):
        super(self, SVD).__init__()

        df = df.values
        df[:2, :] -= 1
        self.raw = df.copy()  # for evaluation
        self.data = df.copy()  # for training
        self.K = K
        self.lambd = lambd
        self.lr_rate = lr_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.user_based = user_based
        self.use_biased = use_biased

        self.n_users = int(np.max(df[:, 0])) + 1 
        self.n_items = int(np.max(df[:, 1])) + 1
        self.n_ratings = len(df)
        self.global_mean = np.mean(df[:, 2])

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


    def normalization(self):
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
                xm = self.X[m, :]
            
                error = xm.dot(Wm) + self.b[m] + dm + self.global_mean - ratings 

                grad_xm = error.dot(Wm.T)/self.n_ratings + self.lambd * xm
                grad_bm = np.sum(error)/self.n_ratings + self.lambd*self.b[m]
                self.X[m, :] -= self.lr_rate*grad_xm.reshape((self.K,))
                self.b_i[m] -= self.learning_rate*grad_bm
        else:
            for m in range(self.n_items):
                user_ids, ratings = self.get_users_who_rate_item(m)

                Wm = self.W[:, user_ids]
                grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
                                               self.lambd*self.X[m, :]
                self.X[m, :] -= self.lr_rate*grad_xm.reshape((self.K,))


    def updateW(self):
        """Backprop and update W using gradient descent
        """
        if self.use_biased:
            for n in range(self.n_users):
                item_ids, ratings = self.get_items_rated_by_user(n)
                Xn = self.X[item_ids, :]
                bn = self.b[item_ids]
                wn = self.W[:, n]

                error = Xn.dot(wn) + bn + self.global_mean + self.d[n] - ratings
                grad_wn = Xn.T.dot(error) / self.n_ratings + self.lambd * wn
                grad_dn = np.sum(error) / self.n_ratings + self.lambd * self.d[n]
                self.W[:, n] -= self.lr_rate * grad_wn.reshape((self.K,))
                self.d_u[n] -= self.lr_rate * grad_dn
        else:
            for n in range(self.n_users):
                item_ids, ratings = self.get_items_rated_by_user(n)
                Xn = self.X[item_ids, :]

                grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
                            self.lambd * self.W[:, n]
                self.W[:, n] -= self.lr_rate*grad_wn.reshape((self.K,))

                
    def fit(self):
        """Training phase
        """
        self.normalization()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if self.verbose & ((it+1) % 10 == 0):
                rmse_train = self.evaluate(self.raw)
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
        ids = np.where(self.data[:, 0] == user_id)[0]
        items_rated_by_u = self.data[ids, 1].tolist()              
        
        y_pred = self.X.dot(self.W[:, user_id])
        predicted_ratings= []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings
    
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