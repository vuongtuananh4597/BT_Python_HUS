import numpy as np
from tensorflow.keras import Sequential, Model, layers
 

class DeepMF(Model):
    """Simple Deep Matrix Factorization
    """
    def __init__(self, n_users, n_items, K):
        super(DeepMF, self).__init__()

        self.pu = Sequential()
        self.pu.add(layers.Embedding(n_users, K, input_length=1))
        self.pu.add(layers.Reshape((K,)))
        
        self.qi = Sequential()
        self.qi.add(layers.Embedding(n_items, K, input_length=1))
        self.qi.add(layers.Reshape((K, )))
        
        self.merge = layers.Dot(axes=1)

    def call(self, inp):
        """Build
        """
        users, items = inp[0], inp[1]
        out_1 = self.pu(users)
        out_2 = self.qi(items)
        out = self.merge([out_1, out_2])
        
        return out
    
    def pred(self, userID, itemID):
        """Prediction given user and item ID
        """
        userID = np.array([[userID]]) - 1
        itemID = np.array([[itemID]]) - 1
        
        return self.predict([userID, itemID])[0][0]