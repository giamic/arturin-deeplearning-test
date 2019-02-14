import os
import numpy as np
import tensorflow as tf
import pickle

import models
from config import DICTIONARY, N_WORDS, TRAIN_FOLDER, VALIDATION_FOLDER, MODEL_FOLDER
from data_generator import DataGenerator

tf.enable_eager_execution()
os.makedirs(MODEL_FOLDER, exist_ok=True)
N_EPOCHS = 5
BATCH_SIZE = 16
SHUFFLE_BUFFER = 1000

with open(os.path.join(TRAIN_FOLDER, 'sequences.pkl'), 'rb') as f:
    seq_trn = pickle.load(f)
with open(os.path.join(TRAIN_FOLDER, 'labels.pkl'), 'rb') as f:
    labels_trn = pickle.load(f)

with open(os.path.join(VALIDATION_FOLDER, 'sequences.pkl'), 'rb') as f:
    seq_vld = pickle.load(f)
with open(os.path.join(VALIDATION_FOLDER, 'labels.pkl'), 'rb') as f:
    labels_vld = pickle.load(f)

# convert the input data to one-hot encoded
training_generator = DataGenerator(seq_trn, labels_trn)
validation_generator = DataGenerator(seq_vld, labels_vld)

model = models.logistic(N_WORDS)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min', baseline=None),
    tf.keras.callbacks.TensorBoard(log_dir=MODEL_FOLDER, histogram_freq=1)
]

model.compile(
    optimizer=tf.train.AdamOptimizer(),  # doesn't work perfectly with keras saver
    # optimizer=tf.keras.optimizers.Adam(),  # doesn't work at all with tf eager execution
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model on dataset
model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    # use_multiprocessing=True,
    # workers=6,
    callbacks=callbacks,
    class_weight={0: 1, 1: 3}  # Mitterand is less common than Chirac, so give it a larger weight
)
# model.fit(
#     seq_trn, labels_trn,
#     epochs=N_EPOCHS,
#     # steps_per_epoch=N_TRAIN // BATCH_SIZE,
#     validation_data=(seq_vld, labels_vld),
#     # validation_steps=N_VALIDATION // BATCH_SIZE,
#     callbacks=callbacks,
#     class_weight={0: 1, 1: 3}  # Mitterand is less common than Chirac, so give it a larger weight
# )
model.summary()

model.save(os.path.join(MODEL_FOLDER, 'model_logistic.h5'))
