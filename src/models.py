import tensorflow as tf


def logistic(input_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=input_dim))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model
