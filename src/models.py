import tensorflow as tf

from config import N_WORDS


def logistic(input_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=input_dim))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model


def lstm():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(None, N_WORDS)))
    model.add(tf.keras.layers.Embedding(N_WORDS+1, 128, mask_zero=True))
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model
