import os
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, initializers, optimizers


ckpt_directory = "/tmp/training_ckpt"
ckpt_prefix = os.path.join(ckpt_directory, "ckpt")


class MLP(Model):
    """Multi-layer Perceptron model
    """
    def __init__(self, mlp_dims, dropout, output_layer=False):
        super().__init__()

        layer = list()
        for mlp_dim in mlp_dims:
            layer.append(layers.Dense(mlp_dim))
            layer.append(layers.BatchNormalization())
            layer.append(layers.Dropout(dropout))
            layer.append(layers.ReLU())
        if output_layer:
            layer.append(layers.Dense(1))

        self.model = keras.Sequential(layer)

    def call(self, x):
        """Build function
        """
        return self.model(x)


class Embedding(Model):
    """Embedding model for feature extraction
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        
        self.embedding = layers.Embedding(sum(field_dims), embed_dim,
                                          embeddings_initializer='glorot_uniform')
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
    def call(self, x):
        
        return self.embedding(x)


class NMF(Model):
    """Neural Matrix Factorization model
    """
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super(NMF, self).__init__()

        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(mlp_dims, dropout, output_layer=False)
        self.fc = layers.Dense(1)

    def call(self, x):
        x = self.embedding(x)
        user_x = x[:, 0]
        item_x = x[:, 1]
        x = self.mlp(tf.reshape(x, [-1, self.embed_output_dim]))
        gmf = user_x * item_x
        x = tf.concat([gmf, x], axis=1)
        x = self.fc(x)
        
        return x
    

class NeuralMF(object):
    """
    """
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, lr_rate):
        super().__init__()
        
        self.model = NMF(field_dims, embed_dim, mlp_dims, dropout)
        self.optim = optimizers.Adam(lr=lr_rate)
        self.ckpt = tf.train.Checkpoint(optimizer=self.optim,
                                             nmf=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_prefix, max_to_keep=5)
        self.num_batch = 0
        
    def _create_batch(self, x, y, n=200, keep_remainder=True):
        """Create batch to feed NeuralMF (because of OOM of AMD's GPU)
        """
        if keep_remainder and (len(x) % n !=0):
            self.num_batch = len(x) // n + 1
        else:
            self.num_batch = len(x) // n
        for i in range(self.num_batch):
            yield x[n*i:n*(i+1)], y[n*i:n*(i+1)]
        
        
    def train(self, x, y, x_valid, y_valid, epochs, n_batch):
        """
        """
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            data_loader = self._create_batch(x, y, n=n_batch, keep_remainder=True)
            for bx, by in tqdm.tqdm(data_loader, total=self.num_batch, desc="Load batch"):
                with tf.GradientTape() as tape:
                    pred = self.model(bx)
                    loss = tf.reduce_mean((pred - by)**2)

                grad = tape.gradient(loss, self.model.trainable_variables)
                self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
            
            if (epoch+1) % 10 == 0:
                valid_pred = self.model(x_valid)
                valid_loss = tf.reduce_mean((pred - y_valid)**2)
                save_path = self.manager.save()
                print("Saved checkpoint for epoch {}: {}".format(epoch+1, save_path))
                print("Epoch {} --- Train loss {:.4f} --- Eval loss {:.4f}".format(epoch+1, loss.numpy(), valid_loss.numpy()))
        
                