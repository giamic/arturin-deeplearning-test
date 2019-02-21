import tensorflow as tf

from config import N_WORDS, TIMESTEPS


def logistic(input_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=input_dim))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model


def lstm(n_emb=128, n_enc=32):
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(None, N_WORDS)))
    model.add(tf.keras.layers.Embedding(N_WORDS + 1, n_emb, mask_zero=True))
    model.add(tf.keras.layers.LSTM(n_enc))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model


def attention(n_emb=128, n_enc=32):
    def f(a):
        return tf.reduce_sum(tf.matmul(a[0], a[1], transpose_a=True), axis=-1)

    x = tf.keras.layers.Input(shape=(TIMESTEPS,))
    emb = tf.keras.layers.Embedding(N_WORDS + 1, n_emb, mask_zero=True)(x)
    # enc_output, enc_state = tf.keras.layers.Bidirectional(
    #     tf.keras.layers.GRU(n_enc, return_sequences=True, return_state=True), merge_mode='concat')(emb)
    enc_output, enc_state = tf.keras.layers.GRU(n_enc, return_sequences=True, return_state=True)(emb)
    dec_state = enc_state
    alignment_input = tf.keras.layers.Concatenate(axis=-1)(
        [tf.keras.layers.RepeatVector(TIMESTEPS)(dec_state), enc_output])
    alignment = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='softmax'))(alignment_input)
    context = tf.keras.layers.Lambda(f)((alignment, enc_output))
    dec_input = tf.keras.layers.Concatenate(axis=-1)([dec_state, context])
    y = tf.keras.layers.Dense(1, activation='sigmoid')(dec_input)
    model = tf.keras.Model(inputs=x, outputs=y)
    return model
