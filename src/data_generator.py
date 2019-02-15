import numpy as np
import tensorflow as tf

from config import N_WORDS, MAX_LENGTH


class LogisticDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, sequences, labels, batch_size, dim=N_WORDS, n_classes=1, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.sequences = sequences
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        seq_temp = [self.sequences[k] for k in indexes]
        X = self.__data_generation(seq_temp)
        y = [int(self.labels[k] == 'M') for k in indexes]

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, seq_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim))

        # Generate data
        for n, s in enumerate(seq_temp):
            for word_index in s:
                X[n][word_index] += 1

        return X


class LSTMDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, sequences, labels, batch_size, dim=N_WORDS, n_classes=1, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.sequences = sequences
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        seqs = [self.sequences[k] for k in indexes]
        timesteps = MAX_LENGTH
        X = np.zeros((self.batch_size, timesteps))
        for i in range(len(seqs)):
            for pos, word in enumerate(seqs[i]):
                X[i, pos] = seqs[i][pos] if word != 0 else N_WORDS

        # timesteps = max([len(s) for s in seqs])
        # X = np.zeros((self.batch_size, timesteps, N_WORDS))
        # for i in range(len(seqs)):
        #     for pos, word in enumerate(seqs[i]):
        #         X[i, pos] = tf.keras.utils.to_categorical(word, N_WORDS)

        y = [int(self.labels[k] == 'M') for k in indexes]

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)
