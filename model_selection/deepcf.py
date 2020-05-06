import tensorflow as tf
from tensorflow.keras import Model, layers


class DeepCF(Model):
    """
    """
    def __init__(self, n_users, n_items, K):
        super(DeepCF, self).__init__()
        
        self.pu = Sequential()
        self.pu.add(layers.Embedding(n_users, K, input_length=1))
        self.pu.add(layers.Reshape(K,))
        
        self.qi = Sequential()
        self.qi.add(layers.Embedding(n_items, K, input_length=1))
        self.qi.add(layers.Reshape(K, ))
        
        self.merge = layers.Merge([self.pu, self.qi], mode='dot', dot_axes=1)
    
    # def call(self, x):

