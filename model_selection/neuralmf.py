from tensorflow.keras import layers, Model


def NeuralMF(num_items, num_users, latent_dim):
    item_input = layers.Input(shape=[1], name='item-input')
    user_input = layers.Input(shape=[1], name='user-input')

    # MLP Embeddings
    item_embedding_mlp = layers.Embedding(num_items + 1, latent_dim, name='movie-embedding-mlp')(item_input)
    item_vec_mlp = layers.Flatten(name='flatten-movie-mlp')(item_embedding_mlp)

    user_embedding_mlp = layers.Embedding(num_users + 1, latent_dim, name='user-embedding-mlp')(user_input)
    user_vec_mlp = layers.Flatten(name='flatten-user-mlp')(user_embedding_mlp)

    # MF Embeddings
    item_embedding_mf = layers.Embedding(num_items + 1, latent_dim, name='movie-embedding-mf')(item_input)
    item_vec_mf = layers.Flatten(name='flatten-movie-mf')(item_embedding_mf)

    user_embedding_mf = layers.Embedding(num_users + 1, latent_dim, name='user-embedding-mf')(user_input)
    user_vec_mf = layers.Flatten(name='flatten-user-mf')(user_embedding_mf)

    # MLP layers
    concat = layers.Concatenate(name='concat')([item_vec_mlp, user_vec_mlp])
    concat_dropout = layers.Dropout(0.2)(concat)
    fc_1 = layers.Dense(100, name='fc-1', activation='relu')(concat_dropout)
    fc_1_bn = layers.BatchNormalization(name='batch-norm-1')(fc_1)
    fc_1_dropout = layers.Dropout(0.2)(fc_1_bn)
    fc_2 = layers.Dense(50, name='fc-2', activation='relu')(fc_1_dropout)
    fc_2_bn = layers.BatchNormalization(name='batch-norm-2')(fc_2)
    fc_2_dropout = layers.Dropout(0.2)(fc_2_bn)

    # Prediction from both layers then concat
    pred_mlp = layers.Dense(10, name='pred-mlp', activation='relu')(fc_2_dropout)
    pred_mf = layers.Dot(name='pred-mf', axes=1)([item_vec_mf, user_vec_mf])
    combine_mlp_mf = layers.Concatenate(name='combine-mlp-mf')([pred_mf, pred_mlp])

    # Last layer
    result = layers.Dense(1, name='result', activation='relu')(combine_mlp_mf)

    model = Model([user_input, item_input], result)
    
    return model