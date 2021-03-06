import os
import pickle

import tensorflow as tf

import models
from config import N_WORDS, TRAIN_FOLDER, VALIDATION_FOLDER, MODEL_FOLDER
from data_generator import LogisticDataGenerator

# tf.enable_eager_execution()
os.makedirs(MODEL_FOLDER, exist_ok=True)
N_EPOCHS = 10
BATCH_SIZE = 32
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
training_generator = LogisticDataGenerator(seq_trn, labels_trn, BATCH_SIZE)
validation_generator = LogisticDataGenerator(seq_vld, labels_vld, BATCH_SIZE)

model = models.logistic(N_WORDS)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min', baseline=None),
    tf.keras.callbacks.TensorBoard(log_dir=MODEL_FOLDER, histogram_freq=1)
]

model.compile(
    # optimizer=tf.train.AdamOptimizer(),  # doesn't work perfectly with keras saver
    optimizer=tf.keras.optimizers.Adam(),  # doesn't work at all with tf eager execution
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model on dataset
model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    epochs=N_EPOCHS,
    callbacks=callbacks,
    class_weight={0: 1, 1: 2}  # Mitterand is less common than Chirac, so give it a larger weight
)

model.summary()

model.save(os.path.join(MODEL_FOLDER, 'model_logistic.h5'))
